#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
import sys
import os
import statistics
import time
from collections import Counter, defaultdict

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    pass 

class SignReaderNode:
    def __init__(self):
        # INITIALIZE NODE
        rospy.init_node('sign_reader_node', anonymous=True)
        
        # COMPETITION CONFIGURATION
        self.TEAM_ID = "TeamRed"
        self.TEAM_PASS = "multi21"
        
        # HARDWARE SETTINGS 
        self.USE_CPU = False 
        self.device = 'cpu' if self.USE_CPU else '0'

        # PATH CONFIGURATION 
        self.sign_model_path = "/home/fizzer/ros_ws/src/controller_pkg/weights/sign.pt" 
        self.char_model_path = "/home/fizzer/ros_ws/src/controller_pkg/weights/best_char.onnx"
        self.topic_name = "/B1/rrbot/camera_wide/image_raw"
        
        # Known Headers
        self.HEADER_TO_ID = {
            "SIZE": 1, "VICTIM": 2, "CRIME": 3, "TIME": 4,
            "PLACE": 5, "MOTIVE": 6, "WEAPON": 7, "BANDIT": 8
        }
        self.VALID_HEADERS = list(self.HEADER_TO_ID.keys())
        
        # Data Aggregation
        self.knowledge_base = defaultdict(Counter)
        self.submitted_facts = set()
        self.CONFIDENCE_THRESHOLD = 3 
        self.game_over = False

        # TIMING & OPTIMIZATION 
        self.frame_count = 0
        self.prev_frame_time = 0 
        
        # Process only 1 out of every 3 frames to allow Sim to run fast
        self.SKIP_FRAMES = 3
        
        # Only read text every 0.3s
        self.last_char_inference = 0
        self.CHAR_INFERENCE_RATE = 0.3 
        
        # Periodic Upload Settings
        self.last_periodic_upload = time.time()
        self.UPLOAD_INTERVAL = 10.0 # Seconds

        # SETUP PUBLISHER
        self.score_pub = rospy.Publisher('/score_tracker', String, queue_size=10)

        # Load Models
        self.sign_model = self.load_model_safe(self.sign_model_path, "Sign")
        self.char_model = self.load_model_safe(self.char_model_path, "Character")

        # WARMUP
        if not self.USE_CPU:
            print("Warming up GPU models...")
            try:
                dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                self.sign_model(dummy_frame, verbose=False, imgsz=320, device=self.device)
                self.char_model(dummy_frame, verbose=False, imgsz=640, device=self.device)
                print("Models warmed up!")
            except Exception as e:
                print(f"Warmup failed: {e}")
                self.USE_CPU = True
                self.device = 'cpu'

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(self.topic_name, Image, self.image_callback, queue_size=1)
        
        print(f"Node Started! Listening to {self.topic_name}...")
        
        rospy.sleep(1.0) 
        self.publish_score(0, "START")

    def publish_score(self, location_id, prediction):
        clean_pred = prediction.replace(" ", "").upper()
        msg = f"{self.TEAM_ID},{self.TEAM_PASS},{location_id},{clean_pred}"
        self.score_pub.publish(msg)
        print(f"SENT SCORE: {msg}")

    def load_model_safe(self, model_path, model_name):
        if not os.path.exists(model_path):
            local_path = os.path.basename(model_path)
            if os.path.exists(local_path):
                model_path = local_path
            else:
                rospy.logerr(f"CRITICAL: Could not find {model_name} model at {model_path}")
                sys.exit(1)
        try:
            return YOLO(model_path, task='detect')
        except Exception as e:
            rospy.logerr(f"Error loading model: {e}")
            sys.exit(1)

    def clean_label(self, label):
        if "sign" in label.lower(): return None
        clean = label.replace("letter_", "").replace("number_", "")
        return clean

    def sort_characters(self, detections, y_threshold=20):
        if not detections: return None, ""
        detections.sort(key=lambda k: (k['bbox'][1] + k['bbox'][3]) / 2)
        lines = []
        current_line = [detections[0]]
        for i in range(1, len(detections)):
            prev_y = (current_line[-1]['bbox'][1] + current_line[-1]['bbox'][3]) / 2
            curr_y = (detections[i]['bbox'][1] + detections[i]['bbox'][3]) / 2
            if abs(curr_y - prev_y) < y_threshold:
                current_line.append(detections[i])
            else:
                lines.append(current_line)
                current_line = [detections[i]]
        lines.append(current_line)

        text_lines = []
        for line in lines:
            line.sort(key=lambda k: k['bbox'][0])
            if len(line) < 2:
                text_lines.append(line[0]['clean_text'])
                continue
            gaps = []
            widths = []
            for i in range(len(line) - 1):
                x2_prev = line[i]['bbox'][2]
                x1_curr = line[i+1]['bbox'][0]
                gap = x1_curr - x2_prev
                gaps.append(max(0, gap))
                widths.append(line[i]['bbox'][2] - line[i]['bbox'][0])
            widths.append(line[-1]['bbox'][2] - line[-1]['bbox'][0])
            median_gap = statistics.median(gaps) if gaps else 0
            avg_width = sum(widths) / len(widths) if widths else 0
            thresh_gap = median_gap * 3.0
            thresh_width = avg_width * 0.6
            line_str = line[0]['clean_text']
            for i, gap in enumerate(gaps):
                if gap > thresh_gap or gap > thresh_width:
                    line_str += " "
                line_str += line[i+1]['clean_text']
            text_lines.append(line_str)

        detected_header = None
        content = ""
        if text_lines:
            potential_header = text_lines[0]
            for valid_h in self.VALID_HEADERS:
                if valid_h in potential_header:
                    detected_header = valid_h
                    break
            if detected_header:
                content = " ".join(text_lines[1:])
            else:
                content = " ".join(text_lines)
        return detected_header, content

    def check_periodic_uploads(self):
        """
        Runs every 10 seconds.
        Submits the best current guess for any clue NOT yet confirmed.
        """
        if time.time() - self.last_periodic_upload > self.UPLOAD_INTERVAL:
            print(f"⏰ {self.UPLOAD_INTERVAL}s TICK: Auto-uploading best guesses...")
            
            for h in self.VALID_HEADERS:
                if h == "BANDIT": continue
                if h in self.submitted_facts: continue
                
                if self.knowledge_base[h]:
                    best_guess, count = self.knowledge_base[h].most_common(1)[0]
                    if h in self.HEADER_TO_ID:
                        loc_id = self.HEADER_TO_ID[h]
                        self.publish_score(loc_id, best_guess)
            
            self.last_periodic_upload = time.time()

    def flush_remaining_clues(self):
        print("BANDIT FOUND! Flushing best guesses...")
        for h in self.VALID_HEADERS:
            if h == "BANDIT": continue
            if h in self.submitted_facts: continue
            if self.knowledge_base[h]:
                best_guess, count = self.knowledge_base[h].most_common(1)[0]
                print(f"FORCING: {h} -> {best_guess}")
                if h in self.HEADER_TO_ID:
                    loc_id = self.HEADER_TO_ID[h]
                    self.publish_score(loc_id, best_guess)
                    self.submitted_facts.add(h)

    def update_knowledge_base(self, header, content):
        if not header or not content: return
        self.knowledge_base[header][content] += 1
        most_common_content, count = self.knowledge_base[header].most_common(1)[0]
        if count >= self.CONFIDENCE_THRESHOLD:
            if header not in self.submitted_facts:
                print(f"CONFIRMED: {header} -> {most_common_content}")
                self.submitted_facts.add(header)
                if header in self.HEADER_TO_ID:
                    loc_id = self.HEADER_TO_ID[header]
                    self.publish_score(loc_id, most_common_content)
                    if header == "BANDIT":
                        self.flush_remaining_clues()
                        self.publish_score(-1, "END")
                        self.game_over = True
                        print("GAME OVER")

    def image_callback(self, data):
        if self.game_over: return
        
        # Periodic Upload Check
        self.check_periodic_uploads()

        # Frame Skipping
        self.frame_count += 1
        if self.frame_count % self.SKIP_FRAMES != 0: return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            try:
                self.process_frame(cv_image)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("GPU OOM")
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
        except CvBridgeError as e:
            print(e)

    def process_frame(self, img):
        vis_img = img.copy()
        
        # FPS Calc
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = curr_time
        
        # Throttling Logic
        run_char_model = (curr_time - self.last_char_inference) > self.CHAR_INFERENCE_RATE

        # Sign Detection
        sign_results = self.sign_model(img, verbose=False, conf=0.5, imgsz=320, device=self.device)
        
        for r in sign_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (x2 - x1) < 60 or (y2 - y1) < 60: continue
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                if not run_char_model: continue

                sign_crop = img[y1:y2, x1:x2]
                if sign_crop.size == 0: continue

                # Character Inference
                char_results = self.char_model(sign_crop, verbose=False, conf=0.45, imgsz=640, device=self.device)
                char_detections = []
                
                for c_res in char_results:
                    for c_box in c_res.boxes:
                        cx1, cy1, cx2, cy2 = map(int, c_box.xyxy[0])
                        cls_id = int(c_box.cls[0])
                        if hasattr(self.char_model, 'names'):
                            raw_label = self.char_model.names[cls_id]
                        else:
                            raw_label = str(cls_id)
                        clean_text = self.clean_label(raw_label)
                        if clean_text:
                            char_detections.append({'clean_text': clean_text, 'bbox': [cx1, cy1, cx2, cy2]})
                            ox1, oy1 = cx1 + x1, cy1 + y1
                            ox2, oy2 = cx2 + x1, cy2 + y1
                            cv2.rectangle(vis_img, (ox1, oy1), (ox2, oy2), (0, 255, 0), 1)

                header, content = self.sort_characters(char_detections)
                if header:
                    self.update_knowledge_base(header, content)
                    cv2.putText(vis_img, f"{header}: {content}", (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        if run_char_model:
            self.last_char_inference = curr_time

        # VISUALIZATION RESIZE
        # Force image to 720p for display to prevent slowdown
        display_img = cv2.resize(vis_img, (640, 360))
        cv2.putText(display_img, f"FPS: {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Robot View", display_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("User pressed q")

if __name__ == '__main__':
    try:
        node = SignReaderNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()