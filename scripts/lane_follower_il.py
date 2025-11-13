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

class PilotNet(nn.Module):
    def __init__(self, out_dim=2):
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
            nn.Linear(64*1*18, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 2),
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

        # Limits & smoothing
        self.max_speed   = float(rospy.get_param("~max_speed", 1.0))         # m/s clamp
        self.max_steer   = float(rospy.get_param("~max_steer", 1.0))         # rad clamp
        self.alpha_speed = float(rospy.get_param("~alpha_speed", 0.3))       # EMA for speed
        self.alpha_steer = float(rospy.get_param("~alpha_steer", 0.5))       # EMA for steer
        self.publish_hz  = float(rospy.get_param("~publish_hz", 15.0))

        # Load preprocess
        with open(prep_path, "r") as f:
            self.prep = json.load(f)

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PilotNet(out_dim=2).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.last_cmd = Twist()
        self.last_pub_time = rospy.get_time()

        self.sub = rospy.Subscriber(img_topic, Image, self.on_image, queue_size=1, buff_size=2**24)

        rospy.loginfo(f"[il] Loaded model from {model_path}")

    def on_image(self, msg: Image):
        # rate limit publishing to ~publish_hz
        now = rospy.get_time()
        if (now - self.last_pub_time) < (1.0 / max(self.publish_hz, 1e-3)):
            return
        self.last_pub_time = now

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn(f"[il] CvBridge error: {e}")
            return

        x = preprocess(cv_img, self.prep)                # numpy
        # ensure channel-first and float32 if not already done inside preprocess()
        if x.ndim == 3 and x.shape[0] != 3:
            x = np.transpose(x, (2,0,1))                 # (3,H,W)
        x = torch.from_numpy(x).unsqueeze(0).to(self.device).float()  # <<< force float32

        with torch.no_grad():
            yhat = self.model(x).cpu().numpy().reshape(-1)  # [2], [steer, speed]
        steer = float(np.clip(yhat[0], -self.max_steer, self.max_steer))

        speed = float(np.clip(yhat[1], -self.max_speed, self.max_speed))

        # EMA smoothing
        self.last_cmd.linear.x  = (1-self.alpha_speed)*self.last_cmd.linear.x  + self.alpha_speed*speed
        self.last_cmd.angular.z = (1-self.alpha_steer)*self.last_cmd.angular.z + self.alpha_steer*steer

        self.cmd_pub.publish(self.last_cmd)

def main():
    rospy.init_node("lane_follower_il")
    # Give ROS graph a moment to wire up (353 docs suggest a short delay)
    time.sleep(1.0)
    LaneFollowerIL()
    rospy.loginfo("[il] Lane follower ready.")
    rospy.spin()

if __name__ == "__main__":
    main()