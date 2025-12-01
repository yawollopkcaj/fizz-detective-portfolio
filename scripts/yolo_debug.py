#!/usr/bin/env python3
import rospy
import torch
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class YoloDebugNode:
    def __init__(self):
        rospy.init_node('yolo_debug_node')
        
        # --- CONFIGURATION ---
        # 1. Path to the YOLOv5 repo (cloned folder)
        self.yolo_repo = "/home/fizzer/yolov5"
        
        # 2. Path to YOUR trained weights
        # Make sure this matches where you copied the file
        self.weights_path = "/home/fizzer/ros_ws/src/controller_pkg/weights/crosswalk_v2.pt" 
        
        # 3. Camera Topic
        self.camera_topic = "/B1/rrbot/camera1/image_raw"
        # ---------------------

        self.bridge = CvBridge()
        
        # Publisher for the image WITH bounding boxes
        self.debug_pub = rospy.Publisher("/yolo/debug_feed", Image, queue_size=1)

        rospy.loginfo(f"Loading YOLO from {self.weights_path}...")
        
        try:
            # Load model locally
            self.model = torch.hub.load(
                self.yolo_repo, 
                'custom', 
                path=self.weights_path, 
                source='local'
            )
            self.model.classes = [0] # Only detect 'Person'
            self.model.conf = 0.25    # Confidence threshold (adjust if needed)
            rospy.loginfo("YOLO Loaded Successfully!")
        except Exception as e:
            rospy.logerr(f"Error loading YOLO: {e}")
            exit(1)

        self.sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        rospy.spin()

    def image_callback(self, msg):
        try:
            # 1. Convert ROS Image to OpenCV
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            # 2. Convert to RGB for YOLO
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            
            # 3. Run Inference
            results = self.model(img_rgb)
            
            # 4. Draw Bounding Boxes
            # results.render() modifies the array in-place, adding boxes
            results.render() 
            
            # 5. Get the annotated image back (it stays in RGB, so convert back to BGR for ROS)
            annotated_img = results.ims[0]
            annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
            
            # 6. Publish the debug image
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(annotated_img_bgr, encoding="bgr8"))
            
        except Exception as e:
            rospy.logwarn(f"Processing error: {e}")

if __name__ == '__main__':
    try:
        YoloDebugNode()
    except rospy.ROSInterruptException:
        pass