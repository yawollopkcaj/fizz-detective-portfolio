#!/usr/bin/env python3
import os, json, time, csv, threading, sys, tty, termios, select
from datetime import datetime
import rospy
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import torch
from torch import nn

class PilotNet(nn.Module):
    # --- UNCHANGED PilotNet DEFINITION ---
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
            nn.Linear(64*9*9, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 1),
        )
    def forward(self, x): return self.fc(self.conv(x))

def preprocess(img, params):
    # --- UNCHANGED preprocess DEFINITION ---
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

class LaneFollowerRecorder:
    def __init__(self):
        self.bridge = CvBridge()
        self.cmd_pub = rospy.Publisher("/B1/cmd_vel", Twist, queue_size=1)

        # --- Mode ---
        self.is_autonomous = True # Start in autonomous mode
        self.lock = threading.Lock()
        self.tty_settings = termios.tcgetattr(sys.stdin) # Save old terminal settings
        rospy.on_shutdown(self.cleanup)

        # === IL Model Params ===
        img_topic = rospy.get_param("~image_topic", "/B1/pi_camera/image_raw")
        model_dir = rospy.get_param("~model_dir", os.path.expanduser("~/il_data/latest/artifacts"))
        model_path = rospy.get_param("~model_path", os.path.join(model_dir, "model.pt"))
        prep_path  = rospy.get_param("~prep_path",  os.path.join(model_dir, "preprocess.json"))

        # Limits & smoothing
        self.max_speed   = float(rospy.get_param("~max_speed", 0.3))
        self.max_steer   = float(rospy.get_param("~max_steer", 2))
        self.alpha_speed = float(rospy.get_param("~alpha_speed", 0.5))
        self.alpha_steer = float(rospy.get_param("~alpha_steer", 0.5))
        self.publish_hz  = float(rospy.get_param("~publish_hz", 10.0))

        # Load preprocess
        with open(prep_path, "r") as f:
            self.prep = json.load(f)

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PilotNet(out_dim=1).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.last_cmd = Twist() # For IL model's EMA smoothing
        self.last_pub_time = rospy.get_time()
        rospy.loginfo(f"[il] Loaded model from {model_path}")

        # === Recorder Params ===
        self.recorder_state = {} # Holds csv_file, writer, out_dir, etc.
        self.last_manual_twist = Twist() # For recording
        self.twist_topic = rospy.get_param("~twist_topic", "/B1/cmd_vel") # Topic to *listen* to for recording
        self.jpg_quality = int(rospy.get_param("~jpg_quality", 95))

        # === ROS Subs ===
        # Subscriber for IL model and Recording
        self.img_sub = rospy.Subscriber(img_topic, Image, self.on_image, queue_size=1, buff_size=2**24)
        # Subscriber for recorder (listens to manual drive commands)
        self.twist_sub = rospy.Subscriber(self.twist_topic, Twist, self.on_twist, queue_size=10)

        # Start keyboard listener
        threading.Thread(target=self.key_monitor_thread, daemon=True).start()

    def key_monitor_thread(self):
        """Monitors for 'p' key press to toggle mode."""
        rospy.loginfo("Key monitor started. Press 'p' to toggle mode, 'Ctrl-C' to quit.")
        try:
            tty.setraw(sys.stdin.fileno())
            while not rospy.is_shutdown():
                if select.select([sys.stdin], [], [], 0.1)[0]: # Poll for 100ms
                    key = sys.stdin.read(1)
                    if key == 'p':
                        self.toggle_mode()
                    elif key == '\x03': # Ctrl-C
                        rospy.loginfo("Ctrl-C pressed, shutting down.")
                        rospy.signal_shutdown("Ctrl-C pressed")
                        break
        except Exception as e:
            rospy.logerr(f"Key monitor error: {e}")
        finally:
            # Restore terminal settings on thread exit
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.tty_settings)

    def toggle_mode(self):
        """Toggles between autonomous and manual recording modes."""
        with self.lock:
            self.is_autonomous = not self.is_autonomous
            if self.is_autonomous:
                rospy.loginfo("Switching to AUTONOMOUS mode.")
                self.stop_recording()
                # Reset EMA filter to prevent sudden jerks
                self.last_cmd = Twist()
            else:
                rospy.loginfo("Switching to MANUAL (recording) mode.")
                self.start_recording()

    def start_recording(self):
        """Initializes a new recording session, same as recorder.py."""
        if self.recorder_state: # Already recording
            return

        # Create output directory
        out_dir = os.path.expanduser("~/il_data/run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        frames_dir = os.path.join(out_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Open CSV
        csv_path = os.path.join(out_dir, "labels.csv")
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["t","steer_rad","v_mps","img_relpath"]) # Write header

        # Write meta.json
        meta = {
            "image_topic": self.img_sub.name,
            "twist_topic": self.twist_sub.name,
            "target_hz": self.publish_hz, # Use the IL publish_hz as the target_hz
            "start_time": time.time()
        }
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Store state
        self.recorder_state = {
            "out_dir": out_dir,
            "frames_dir": frames_dir,
            "csv_file": csv_file,
            "writer": writer
        }
        rospy.loginfo(f"[recorder] Saving to {out_dir}")

    def stop_recording(self):
        """Finalizes and closes the current recording session."""
        if self.recorder_state and self.recorder_state.get("csv_file"):
            try:
                self.recorder_state["csv_file"].close()
                rospy.loginfo(f"[recorder] Stopped recording. Data saved to {self.recorder_state['out_dir']}")
            except Exception as e:
                rospy.logerr(f"Error closing CSV file: {e}")
        self.recorder_state = {}

    def on_twist(self, msg: Twist):
        """Callback for manual twist commands. Only used for recording."""
        # Store the latest manual command
        self.last_manual_twist = msg

    def on_image(self, msg: Image):
        """
        Main callback.
        Handles EITHER IL inference OR data recording, based on mode.
        """
        # Rate limit publishing/recording
        now = rospy.get_time()
        if (now - self.last_pub_time) < (1.0 / max(self.publish_hz, 1e-3)):
            return
        self.last_pub_time = now

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn(f"[il] CvBridge error: {e}")
            return

        # Check current mode
        with self.lock:
            is_auto = self.is_autonomous

        if is_auto:
            # --- AUTONOMOUS MODE ---
            # Preprocess image
            x = preprocess(cv_img, self.prep)
            if x.ndim == 3 and x.shape[0] != 3:
                x = np.transpose(x, (2,0,1))
            x = torch.from_numpy(x).unsqueeze(0).to(self.device).float()

            # Run model
            with torch.no_grad():
                yhat = self.model(x).cpu().numpy().reshape(-1)
            steer = float(np.clip(yhat[0], -self.max_steer, self.max_steer))
            speed = self.max_speed

            # EMA smoothing
            self.last_cmd.linear.x  = (1-self.alpha_speed)*self.last_cmd.linear.x  + self.alpha_speed*speed
            self.last_cmd.angular.z = (1-self.alpha_steer)*self.last_cmd.angular.z + self.alpha_steer*steer

            # Publish command
            self.cmd_pub.publish(self.last_cmd)

        else:
            # --- MANUAL RECORDING MODE ---
            if not self.recorder_state.get("writer"):
                rospy.logwarn_throttle(1.0, "[recorder] In manual mode but recorder not initialized.")
                return

            # Save image frame
            idx = int((now % 1e9) * 1e3)  # quasi-unique
            relname = f"frames/frame_{idx:09d}.jpg"
            abspath = os.path.join(self.recorder_state['out_dir'], relname)
            cv2.imwrite(abspath, cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_quality])

            # Get latest manual command
            steer = float(self.last_manual_twist.angular.z)
            v     = float(self.last_manual_twist.linear.x)

            # Write to CSV
            self.recorder_state['writer'].writerow([f"{now:.6f}", f"{steer:.6f}", f"{v:.6f}", relname])
            self.recorder_state['csv_file'].flush()

            # NOTE: We DO NOT publish to cmd_vel here.
            # Your separate manual controller (joystick/teleop) is doing that.
            # This script just listens.

    def cleanup(self):
        """Called on node shutdown to restore terminal and close files."""
        rospy.loginfo("Restoring terminal settings...")
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.tty_settings)
        self.stop_recording() # Ensure file is closed
        # Send a final zero-velocity command
        rospy.loginfo("Sending stop command.")
        self.cmd_pub.publish(Twist())

def main():
    rospy.init_node("lane_follower_recorder")
    time.sleep(1.0) # Give ROS graph a moment

    # Save terminal settings before we modify them
    tty_settings = termios.tcgetattr(sys.stdin)

    try:
        LaneFollowerRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutdown requested.")
    except Exception as e:
        rospy.logerr(f"Unhandled exception in main: {e}")
    finally:
        # Ensure terminal settings are restored on exit
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, tty_settings)
        rospy.loginfo("[il_recorder] Shutting down and restoring terminal.")

if __name__ == "__main__":
    main()