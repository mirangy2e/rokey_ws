import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2
import threading
import time
import os
import shutil
import sys
import csv
import json
from ultralytics import YOLO
from pathlib import Path
import torch

# ================================
ROBOT_NAMESPACE = 'robot4'
RGB_TOPIC = f'/{ROBOT_NAMESPACE}/oakd/rgb/preview/image_raw'
DEPTH_TOPIC = f'/{ROBOT_NAMESPACE}/oakd/stereo/image_raw'
CAMERA_INFO_TOPIC = f'/{ROBOT_NAMESPACE}/oakd/stereo/camera_info'

RESIZE_WIDTH = 320   # 입력 이미지 축소 사이즈
RESIZE_HEIGHT = 320
# ================================

class YoloDepthProcessor(Node):
    def __init__(self, model, output_dir):
        super().__init__('yolo_depth_processor')
        self.model = model
        self.output_dir = output_dir
        self.bridge = CvBridge()

        self.classNames = self.model.names if hasattr(self.model, 'names') else []

        self.K = None
        self.latest_rgb = None
        self.latest_depth = None
        self.processed_frame = None
        self.lock = threading.Lock()
        self.should_shutdown = False

        self.csv_output = []
        self.max_object_count = 0
        self.confidences = []

        # ✅ queue_size=1
        self.create_subscription(Image, RGB_TOPIC, self.rgb_callback, 1)
        self.create_subscription(Image, DEPTH_TOPIC, self.depth_callback, 1)
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, 1)

    def camera_info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(f"CameraInfo received: fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}")

    def rgb_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.latest_rgb = img
        except Exception as e:
            self.get_logger().error(f"RGB image conversion failed: {e}")

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.lock:
                self.latest_depth = depth
        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {e}")

    def inference_loop(self):
        while rclpy.ok() and not self.should_shutdown:
            with self.lock:
                rgb = self.latest_rgb
                depth = self.latest_depth
                K = self.K

            if rgb is None or depth is None or K is None:
                time.sleep(0.005)
                continue

            # ➡️ 이미지 downscale
            rgb_resized = cv2.resize(rgb, (RESIZE_WIDTH, RESIZE_HEIGHT))

            # ➡️ YOLO 추론 (stream=True, iou=0.6)
            results = self.model.predict(rgb_resized, stream=True, iou=0.6)

            frame = rgb.copy()
            scale_x = rgb.shape[1] / RESIZE_WIDTH
            scale_y = rgb.shape[0] / RESIZE_HEIGHT
            object_count = 0

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    x_center, y_center, _, _ = box.xywh[0].cpu().numpy()
                    u = int(x_center * scale_x)
                    v = int(y_center * scale_y)

                    if not (0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]):
                        continue

                    z_mm = float(depth[v, u])
                    z = z_mm / 1000.0

                    fx = K[0, 0]
                    fy = K[1, 1]
                    cx = K[0, 2]
                    cy = K[1, 2]

                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy

                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)

                    confidence = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = self.classNames[cls] if cls < len(self.classNames) else f"class_{cls}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (u, v), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"{label} {confidence:.2f} {z:.2f}m", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    self.csv_output.append([label, confidence, u, v, x, y, z])
                    self.confidences.append(confidence)
                    object_count += 1

            self.max_object_count = max(self.max_object_count, object_count)

            with self.lock:
                self.processed_frame = frame

            time.sleep(0.005)

    def save_output(self):
        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, 'output.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Label', 'Confidence', 'Pixel_u', 'Pixel_v', 'X_m', 'Y_m', 'Z_m'])
            writer.writerows(self.csv_output)

        with open(os.path.join(self.output_dir, 'output.json'), 'w') as f:
            json.dump(self.csv_output, f, indent=4)

        with open(os.path.join(self.output_dir, 'statistics.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Max Object Count', 'Average Confidence'])
            avg_conf = sum(self.confidences) / len(self.confidences) if self.confidences else 0
            writer.writerow([self.max_object_count, avg_conf])

def main():
    model_path = input("Enter path to model file (.pt, .onnx, .engine): ").strip()

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    suffix = Path(model_path).suffix.lower()
    if suffix == '.pt':
        model = YOLO(model_path)
    elif suffix in ['.onnx', '.engine']:
        model = YOLO(model_path, task='detect')
    else:
        print(f"Unsupported model format: {suffix}")
        sys.exit(1)

    if torch.cuda.is_available():
        model.to('cuda')
        model.half()  # ➡️ FP16 모드
        print("Using GPU (FP16) for inference.")
    else:
        print("GPU not available. Using CPU.")

    output_dir = './output'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    rclpy.init()
    node = YoloDepthProcessor(model, output_dir)

    try:
        spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        inference_thread = threading.Thread(target=node.inference_loop, daemon=True)

        spin_thread.start()
        inference_thread.start()

        while rclpy.ok() and not node.should_shutdown:
            frame = None
            with node.lock:
                if node.processed_frame is not None:
                    frame = node.processed_frame.copy()

            if frame is not None:
                cv2.imshow("YOLO + Depth", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                node.get_logger().info("Shutdown requested by user.")
                node.should_shutdown = True
                break

            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_output()
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        print("Shutdown complete.")
        sys.exit(0)

if __name__ == '__main__':
    main()
