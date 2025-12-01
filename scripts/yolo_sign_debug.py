#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

class YoloDebugNode:
    def __init__(self):
        rospy.init_node('yolo_debug_node')
        
        # --- CONFIGURATION ---
        # 1. Path to YOUR trained weights (YOLOv8 model)
        self.weights_path = "/home/fizzer/ros_ws/src/controller_pkg/weights/best.pt" 
        
        # 2. Camera Topic
        self.camera_topic = "/B1/rrbot/camera1/image_raw"
        
        # 3. Settings
        self.conf_thresh = 0.25
        # ---------------------

        self.bridge = CvBridge()
        
        # Publisher for the image WITH bounding boxes
        self.debug_pub = rospy.Publisher("/yolo/debug_feed", Image, queue_size=1)

        rospy.loginfo(f"Loading YOLOv8 from {self.weights_path}...")
        
        try:
            # Load YOLOv8 model directly
            self.model = YOLO(self.weights_path)
            rospy.loginfo("✅ YOLOv8 Loaded Successfully!")
        except Exception as e:
            rospy.logerr(f"❌ Error loading YOLO: {e}")
            exit(1)

        self.sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        rospy.spin()

    def image_callback(self, msg):
        try:
            # 1. Convert ROS Image to OpenCV
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            # 2. Run Inference
            # YOLOv8 handles the color conversion internally, but explicit conversion is safer
            results = self.model.predict(cv_img, conf=self.conf_thresh, verbose=False)
            
            # 3. Draw Bounding Boxes
            # .plot() returns the image as a BGR numpy array with boxes drawn
            annotated_frame = results[0].plot()
            
            # 4. Publish the debug image
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8"))
            
            # Optional: Print detection count to terminal
            det_count = len(results[0].boxes)
            if det_count > 0:
                rospy.loginfo(f"Detections: {det_count}")
            
        except Exception as e:
            rospy.logwarn(f"Processing error: {e}")

if __name__ == '__main__':
    try:
        YoloDebugNode()
    except rospy.ROSInterruptException:
        pass