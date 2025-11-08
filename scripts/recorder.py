#!/usr/bin/env python3
import rospy, os, csv, time
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2

class Recorder:
    def __init__(self):
        self.out_dir = rospy.get_param("~out_dir", os.path.expanduser("~/il_data/run_%d" % int(time.time())))
        self.img_topic = rospy.get_param("~image", "/B1/pi_camera/image_raw")
        self.twist_topic = rospy.get_param("~twist", "/B1/cmd_vel")
        os.makedirs(os.path.join(self.out_dir, "frames"), exist_ok=True)
        self.bridge = CvBridge()
        self.last_twist = Twist()
        self.csv = open(os.path.join(self.out_dir, "labels.csv"), "w", newline="")
        self.w = csv.writer(self.csv); self.w.writerow(["t","steer_rad","v_mps","img_relpath"])
        rospy.Subscriber(self.img_topic, Image, self.on_img, queue_size=1)
        rospy.Subscriber(self.twist_topic, Twist, self.on_twist, queue_size=1)
        rospy.loginfo("Recording to %s", self.out_dir)

    def on_twist(self, tw): self.last_twist = tw

    def on_img(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        ts = msg.header.stamp.to_sec() if msg.header.stamp else rospy.get_time()
        name = f"frames/{ts:.3f}.jpg"
        cv2.imwrite(os.path.join(self.out_dir, name), img)
        self.w.writerow([f"{ts:.3f}", self.last_twist.angular.z, self.last_twist.linear.x, name])

if __name__ == "__main__":
    rospy.init_node("il_recorder")
    Recorder()
    rospy.spin()