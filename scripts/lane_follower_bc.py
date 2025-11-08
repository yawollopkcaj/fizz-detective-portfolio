#!/usr/bin/env python3
import os, json, cv2, numpy as np, rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge

USE_TFLITE = True
try:
    from tensorflow.lite.python.interpreter import Interpreter
except Exception:
    USE_TFLITE = False
try:
    import tensorflow as tf
except Exception:
    pass

class ScoreClient:
    def __init__(self, team, password):
        self.pub = rospy.Publisher("/score_tracker", String, queue_size=1)
        self.team, self.pw = team, password
        self.started = False
    def start(self):
        if not self.started:
            self.pub.publish(String(f"{self.team},{self.pw},0,NA"))
            self.started = True
    def stop(self):
        if self.started:
            self.pub.publish(String(f"{self.team},{self.pw},-1,NA"))

class ILController:
    def __init__(self):
        # params
        self.team = rospy.get_param("~team", "TeamRed")
        self.password = rospy.get_param("~pass", "multi21")
        self.image_topic = rospy.get_param("~image_topic", "/B1/rrbot/camera1/image_raw")
        self.cmd_topic   = rospy.get_param("~cmd_vel_topic", "/B1/cmd_vel")
        self.bc_dir = rospy.get_param("~bc_dir", os.path.join(os.path.dirname(__file__), "..", "models", "bc"))
        self.v_lin = float(rospy.get_param("~v_lin", 0.22))
        self.max_omega = float(rospy.get_param("~max_omega", 1.0))
        self.rate_hz = float(rospy.get_param("~rate_hz", 12.0))

        # meta
        meta_path = os.path.join(self.bc_dir, "meta.json")
        meta = json.load(open(meta_path))
        self.img_w, self.img_h = int(meta["img_w"]), int(meta["img_h"])
        self.crop_top = float(meta["crop_top"])

        # model
        self.session = None
        self.in_idx = self.out_idx = None
        if USE_TFLITE and os.path.exists(os.path.join(self.bc_dir, "bc.tflite")):
            model_path = os.path.join(self.bc_dir, "bc.tflite")
            self.session = Interpreter(model_path=model_path)
            self.session.allocate_tensors()
            self.in_idx  = self.session.get_input_details()[0]["index"]
            self.out_idx = self.session.get_output_details()[0]["index"]
            rospy.loginfo("Loaded TFLite model: %s", model_path)
        else:
            model_path = os.path.join(self.bc_dir, "saved_model")
            self.session = tf.keras.models.load_model(model_path)
            rospy.loginfo("Loaded Keras SavedModel: %s", model_path)

        # io
        self.bridge = CvBridge()
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.sub = rospy.Subscriber(self.image_topic, Image, self.on_img, queue_size=1)
        self.clock = rospy.Rate(self.rate_hz)

        # scoring
        self.score = ScoreClient(self.team, self.password)
        rospy.sleep(1.0)  # give ROS time to stabilize
        rospy.loginfo("ILController ready: img=%s cmd=%s v=%.2f", self.image_topic, self.cmd_topic, self.v_lin)

    def infer(self, crop):
        x = crop.astype(np.float32) / 255.0
        if USE_TFLITE and isinstance(self.session, Interpreter):
            x = np.expand_dims(x, 0)  # 1,H,W,3
            self.session.set_tensor(self.in_idx, x)
            self.session.invoke()
            steer = float(self.session.get_tensor(self.out_idx)[0,0])
        else:
            x = np.expand_dims(x, 0)  # 1,H,W,3
            steer = float(self.session(x, training=False).numpy()[0,0])
        return steer

    def on_img(self, msg):
        self.score.start()
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w, _ = img.shape
        y0 = int(h * self.crop_top)
        crop = img[y0:h, :]
        crop = cv2.resize(crop, (self.img_w, self.img_h))
        omega = self.infer(crop)
        omega = float(np.clip(omega, -self.max_omega, self.max_omega))

        tw = Twist()
        tw.linear.x = self.v_lin
        tw.angular.z = omega
        self.cmd_pub.publish(tw)
        self.clock.sleep()

    def shutdown(self):
        self.cmd_pub.publish(Twist())
        self.score.stop()

if __name__ == "__main__":
    rospy.init_node("lane_follower_bc")
    node = ILController()
    rospy.on_shutdown(node.shutdown)
    rospy.spin()