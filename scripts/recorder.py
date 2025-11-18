#!/usr/bin/env python3
import os, csv, time, argparse, json
from datetime import datetime
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2

class Recorder:
    """
    Records (image, Twist) pairs for imitation learning.
    Writes:
      <out_dir>/labels.csv with columns: t, steer_rad, v_mps, img_relpath
      <out_dir>/frames/frame_000001.jpg ...
    """
    def __init__(self):
        self.bridge = CvBridge()
        self.last_twist = Twist()
        self.last_img_time = 0.0

        # ROS params (override in launch if needed)
        self.img_topic   = rospy.get_param("~image_topic", "/B1/rrbot/camera1/image_raw")
        self.twist_topic = rospy.get_param("~twist_topic", "/B1/cmd_vel")
        self.target_hz   = float(rospy.get_param("~target_hz", 10.0)) # throttle image saves
        self.jpg_quality = int(rospy.get_param("~jpg_quality", 95))

        # Output directory
        default_dir = os.path.expanduser("~/il_data/run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.out_dir = rospy.get_param("~out_dir", default_dir)
        self.frames_dir = os.path.join(self.out_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)

        # CSV
        self.csv_path = os.path.join(self.out_dir, "labels.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(["t","steer_rad","v_mps","img_relpath"])

        # Meta
        meta = {
            "image_topic": self.img_topic,
            "twist_topic": self.twist_topic,
            "target_hz": self.target_hz,
            "start_time": time.time()
        }
        with open(os.path.join(self.out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Subs
        rospy.Subscriber(self.twist_topic, Twist, self.on_twist, queue_size=10)
        rospy.Subscriber(self.img_topic, Image, self.on_image, queue_size=1)

        rospy.loginfo(f"[recorder] Saving to {self.out_dir}")

    def on_twist(self, msg: Twist):
        self.last_twist = msg

    def on_image(self, msg: Image):
        now = rospy.get_time()
        if self.target_hz > 0:
            min_dt = 1.0 / self.target_hz
            if (now - self.last_img_time) < min_dt:
                return
        self.last_img_time = now

        # Convert & save
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn(f"[recorder] CvBridge error: {e}")
            return

        idx = int((now % 1e9) * 1e3)  # quasi-unique in a run
        relname = f"frames/frame_{idx:09d}.jpg"
        abspath = os.path.join(self.out_dir, relname)
        cv2.imwrite(abspath, cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_quality])

        steer = float(self.last_twist.angular.z)
        v     = float(self.last_twist.linear.x)

        self.writer.writerow([f"{now:.6f}", f"{steer:.6f}", f"{v:.6f}", relname])
        self.csv_file.flush()

def main():
    rospy.init_node("il_recorder")
    Recorder()
    rospy.loginfo("[recorder] Ready. Drive with your expert controller to collect data.")
    rospy.spin()

if __name__ == "__main__":
    main()