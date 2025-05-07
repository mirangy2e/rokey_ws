import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs

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
RGB_TOPIC = f'/{ROBOT_NAMESPACE}/cropped/rgb/image_raw'
DEPTH_TOPIC = f'/{ROBOT_NAMESPACE}/cropped/depth/image_raw'
CAMERA_INFO_TOPIC = f'/{ROBOT_NAMESPACE}/cropped/camera_info'
MAP_FRAME = f'{ROBOT_NAMESPACE}/map'
# ================================

class YoloDepthToMap(Node):
    def __init__(self, model, output_dir):
        super().__init__('yolo_depth_to_map')
        self.model = model
        self.output_dir = output_dir
        self.bridge = CvBridge()

        self.classNames = self.model.names if hasattr(self.model, 'names') else []

        self.K = None
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_rgb_msg = None
        self.processed_frame = None
        self.lock = threading.Lock()
        self.should_shutdown = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.csv_output = []
        self.max_object_count = 0
        self.confidences = []

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
                self.latest_rgb_msg = msg
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
                rgb_msg = self.latest_rgb_msg

            if rgb is None or depth is None or K is None or rgb_msg is None:
                time.sleep(0.005)
                continue

            frame = rgb.copy()
            results = self.model(frame)
            object_count = 0

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    x_center, y_center, _, _ = box.xywh[0].cpu().numpy()
                    u, v = int(x_center), int(y_center)

                    if not (0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]):
                        continue

                    z_mm = float(depth[v, u])
                    z = z_mm / 1000.0  # mm → m 변환

                    fx, fy = K[0, 0], K[1, 1]
                    cx, cy = K[0, 2], K[1, 2]
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy

                    # TF 변환
                    try:
                        pt_camera = PointStamped()
                        pt_camera.header.frame_id = rgb_msg.header.frame_id
                        pt_camera.header.stamp = rclpy.time.Time().to_msg()
                        pt_camera.point.x = x
                        pt_camera.point.y = y
                        pt_camera.point.z = z

                        try:
                            pt_map = self.tf_buffer.transform(
                                pt_camera, MAP_FRAME,
                                timeout=rclpy.duration.Duration(seconds=0.5)
                            )
                            map_x, map_y, map_z = pt_map.point.x, pt_map.point.y, pt_map.point.z
                            self.get_logger().info(
                                f"[{pt_camera.header.frame_id}] ({x:.2f}, {y:.2f}, {z:.2f}) → "
                                f"[{MAP_FRAME}] ({map_x:.2f}, {map_y:.2f}, {map_z:.2f})"
                            )
                        except Exception as e:
                            self.get_logger().warn(f"[TF] 변환 실패 ({pt_camera.header.frame_id} → {MAP_FRAME}): {e}")
                            map_x, map_y, map_z = float('nan'), float('nan'), float('nan')

                    except Exception as e:
                        self.get_logger().warn(f"[TF] 예외 발생: {e}")
                        map_x, map_y, map_z = float('nan'), float('nan'), float('nan')

                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    confidence = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = self.classNames[cls] if cls < len(self.classNames) else f"class_{cls}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (u, v), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"{label} {confidence:.2f} {z:.2f}m", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    self.csv_output.append([label, confidence, u, v, x, y, z, map_x, map_y, map_z])
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
            writer.writerow(['Label', 'Confidence', 'Pixel_u', 'Pixel_v', 'X_m', 'Y_m', 'Z_m', 'Map_X', 'Map_Y', 'Map_Z'])
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
        print("Using GPU for inference.")
    else:
        print("GPU not available. Using CPU.")

    output_dir = './output'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    rclpy.init()
    node = YoloDepthToMap(model, output_dir)

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
                cv2.imshow("YOLO + Depth + Map", frame)

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
