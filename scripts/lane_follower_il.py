#!/usr/bin/env python3
import os, json, time
import rospy
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import torch
from torch import nn

# State Machine Constants
DRIVING_STATE = 0
STOPPED_STATE = 1
WAITING_STATE = 2
COOLDOWN_STATE = 3

class PilotNet(nn.Module):
    def __init__(self, out_dim=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ELU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*9*9,100), nn.ELU(),
            nn.Dropout(0.5), # <--- ADD THIS
            nn.Linear(100, 50), nn.ELU(),
            nn.Dropout(0.5), # <--- ADD THIS
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, out_dim),
        )
    def forward(self, x): return self.fc(self.conv(x))

def preprocess(img, params):
    h = img.shape[0]
    top = int(h * params["crop_top"])
    bottom = int(h * (1.0 - params["crop_bottom"]))
    img = img[top:bottom]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out_w, out_h = params["resize"]
    img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    mean = np.array(params["mean"])
    std  = np.array(params["std"]) + 1e-6
    img = (img - mean) / std
    img = np.transpose(img, (2,0,1))  # CHW
    return img

class LaneFollowerIL:
    def __init__(self):
        self.bridge = CvBridge()
        self.cmd_pub = rospy.Publisher("/B1/cmd_vel", Twist, queue_size=1)

        # Model paths
        img_topic = rospy.get_param("~image_topic", "/B1/pi_camera/image_raw")
        model_dir = rospy.get_param("~model_dir", os.path.expanduser("~/il_data/latest/artifacts"))
        model_path = rospy.get_param("~model_path", os.path.join(model_dir, "model.pt"))
        prep_path  = rospy.get_param("~prep_path",  os.path.join(model_dir, "preprocess.json"))
        yolo_path = rospy.get_param("~yolo_model_path", os.path.join(model_dir, "yolov5s.pt"))

        # Limits & smoothing
        self.max_speed   = float(rospy.get_param("~max_speed", 0.3))         # m/s clamp
        self.max_steer   = float(rospy.get_param("~max_steer", 2))         # rad clamp
        self.alpha_speed = float(rospy.get_param("~alpha_speed", 0.5))       # EMA for speed
        self.alpha_steer = float(rospy.get_param("~alpha_steer", 0.5)) # (inverse) how much robot remembers previous speed
        self.steer_gain = float(rospy.get_param("~steer_gain", 1.0))
        self.publish_hz  = float(rospy.get_param("~publish_hz", 30.0))

        # Crosswalk and YOLO params
        self.red_pixel_thresh = 20000 # Thresh to trigger stop
        self.pedestrian_debounce = 0.5  # seconds
        self.crosswalk_cooldown = 5.0  # seconds
        self.time_since_last_seen = 0.0
        self.manual_override = False

        # Load preprocess
        with open(prep_path, "r") as f:
            self.prep = json.load(f)

        # Load IL model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PilotNet(out_dim=1).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Load YOLO model for crosswalk detection
        rospy.loginfo(f"[il] Loading YOLO model from {yolo_path}...")
        try:
            self.yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_path)
            
            # Option B: Offline (If you have cloned the yolov5 repo to your robot)
            # self.yolo = torch.hub.load('/home/fizzer/yolov5', 'custom', path=yolo_path, source='local')
            
            self.yolo.to(self.device)
            
           # Make sure pedestrian class is at index 0
            self.yolo.classes = [0] 
            
            rospy.loginfo("[il] YOLO model loaded successfully.")
        except Exception as e:
            rospy.logerr(f"[il] Failed to load YOLO: {e}")

        # State variables for crosswalk handling
        self.state = DRIVING_STATE
        self.last_crosswalk_time = 0

        self.last_cmd = Twist()
        self.last_pub_time = rospy.get_time()

        self.sub = rospy.Subscriber(img_topic, Image, self.on_image, queue_size=1, buff_size=2**24)

        rospy.loginfo(f"[il] Loaded model from {model_path}")

    # Helper method to detect red crosswalk
    def detet_red_crosswalk(self, cv_img):
        # Convert to HSV
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

        # Define red range
        lower1 = np.array([0, 150, 100]); upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 150, 100]); upper2 = np.array([180, 255, 255])

        mask = cv2.inRange(hsv, lower1, upper1) + cv2.inRange(hsv, lower2, upper2)

        # Look only at bottom half of screen
        h,w, _= cv_img.shape
        roi = mask[int(h/w):, :]

        count = cv2.countNonZero(roi)
        return count > self.red_pixel_thresh
    
    # Helper method to detect pedestrians using YOLO
    def detect_pedestrian(self, cv_img):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        results = self.yolo(img_rgb)

        detections = results.xyxy[0].cpu().numpy() # outputs (x1, y1, x2, y2, conf, cls)
        for det in detections:
            conf = det[4]
            cls = int(det[5])
            if cls == 0 and conf > 0.1: # class 0 is pedestrian
                return True
        return False
            
    # Helper method to stop the robot
    def stop_robot(self):
        stop_cmd = Twist() # Defult twist is all zeros
        self.cmd_pub.publish(stop_cmd)
        self.last_cmd = stop_cmd

    # Driving Logic
    def drive_with_il(self, cv_img):
        x = preprocess(cv_img, self.prep)
        if x.ndim == 3 and x.shape[0] != 3:
            x = np.transpose(x, (2,0,1))
        x = torch.from_numpy(x).unsqueeze(0).to(self.device).float()

        with torch.no_grad():
            yhat = self.model(x).cpu().numpy().reshape(-1)
        
        steer = float(np.clip(yhat[0], -self.max_steer, self.max_steer))
        steer *= self.steer_gain
        speed = self.max_speed 

        # EMA smoothing
        self.last_cmd.linear.x  = (1-self.alpha_speed)*self.last_cmd.linear.x  + self.alpha_speed*speed
        self.last_cmd.angular.z = (1-self.alpha_steer)*self.last_cmd.angular.z + self.alpha_steer*steer
        self.cmd_pub.publish(self.last_cmd)

    # Main loop with state machine
    def on_image(self, msg: Image):
        # rate limit publishing to ~publish_hz
        now = rospy.get_time()
        if (now - self.last_pub_time) < (1.0 / max(self.publish_hz, 1e-3)):
            return
        self.last_pub_time = now

        # CV Bridge conversion
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn(f"[il] CvBridge error: {e}")
            return
        
        debug_img = cv_img.copy()

        # Display the window so 'p' works BEFORE STARTING TO LINE FOLLOW
        try:
            cv2.imshow("Robot View", debug_img)
        except: pass

        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            rospy.loginfo("[il] MANUAL OVERRIDE TRIGGERED (Key 'p')")
            self.manual_override = True

        # BEGIN STATE MACHINE LOGIC
        if self.state == DRIVING_STATE:
            self.drive_with_il(cv_img)

            # Check for red crosswalk
            if self.detet_red_crosswalk(cv_img):
                rospy.loginfo("[il] Red crosswalk detected. Stopping.")
                self.state = STOPPED_STATE
                self.stop_robot()
            
        elif self.state == STOPPED_STATE:
            self.stop_robot()

            if self.manual_override:
                rospy.loginfo("[il] Override accepted. Moving to cooldown.")
                self.state = WAITING_STATE
                self.last_crosswalk_time = now

            # Run YOLO to check for pedestrians
            elif self.detect_pedestrian(cv_img):
                rospy.loginfo("[il] Pedestrian detected in crosswalk. Waiting...")
                self.time_since_last_seen = now  # Reset wait time
                self.state = WAITING_STATE

        elif self.state == WAITING_STATE:
            self.stop_robot()

            if self.detect_pedestrian(cv_img):
                # Still seeing pedestrian, reset wait timer
                self.time_since_last_seen = now
            else:
                # If clear, wait for debounce period then proceed
                if now - self.time_since_last_seen > self.pedestrian_debounce:
                    rospy.loginfo("[il] Pedestrian clear. Resuming.")
                    self.state = COOLDOWN_STATE
                    self.last_crosswalk_time = now

        elif self.state == COOLDOWN_STATE:
            self.drive_with_il(cv_img)

            if (now - self.last_crosswalk_time) > self.crosswalk_cooldown:
                rospy.loginfo("[il] Cooldown complete. Resuming normal driving.")
                self.state = DRIVING_STATE

# Main entry point
def main():
    rospy.init_node("lane_follower_il")
    time.sleep(5.0) # Give ROS graph a moment to wire up
    LaneFollowerIL()
    rospy.loginfo("[il] Lane follower ready.")
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()