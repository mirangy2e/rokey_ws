import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
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
import sys
from ultralytics import YOLO
from pathlib import Path
import torch
import argparse

RGB_TOPIC = 'cropped/rgb/image_raw'
DEPTH_TOPIC = 'cropped/depth/image_raw'
CAMERA_INFO_TOPIC = 'cropped/camera_info'

class YoloDepthToMap(Node):
    def __init__(self, model):
        super().__init__('yolo_depth_to_map')
        self.get_logger().info("YoloDepthToMap node is starting...")

        self.model = model
        self.bridge = CvBridge()
        self.classNames = getattr(self.model, 'names', [])

        self.K = None
        self.latest_rgb = self.latest_depth = self.latest_rgb_msg = None
        self.overlay_info = []
        self.lock = threading.Lock()
        self.should_shutdown = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(Image, RGB_TOPIC, self.rgb_callback, 1)
        self.create_subscription(Image, DEPTH_TOPIC, self.depth_callback, 1)
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, 1)

        self.create_timer(0.03, self.inference_callback)  # 30Hz 추론 루프

    def camera_info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(f"CameraInfo: fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}")

    def rgb_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self.lock:
                self.latest_rgb, self.latest_rgb_msg = img, msg
        except Exception as e:
            self.get_logger().error(f"RGB conversion error: {e}")

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            with self.lock:
                self.latest_depth = depth
        except Exception as e:
            self.get_logger().error(f"Depth conversion error: {e}")

    def transform_to_map(self, pt_camera: PointStamped, class_name: str):
        try:
            pt_map = self.tf_buffer.transform(pt_camera, 'map', timeout=rclpy.duration.Duration(seconds=0.5))
            x, y, z = pt_map.point.x, pt_map.point.y, pt_map.point.z
            self.get_logger().info(f"[TF] {class_name} → map: (x={x:.2f}, y={y:.2f}, z={z:.2f})")
            return x, y, z
        except Exception as e:
            self.get_logger().warn(f"[TF] class={class_name} 변환 실패: {e}")
            return float('nan'), float('nan'), float('nan')

    def inference_callback(self):
        with self.lock:
            rgb, depth, K, rgb_msg = self.latest_rgb, self.latest_depth, self.K, self.latest_rgb_msg

        if any(v is None for v in (rgb, depth, K, rgb_msg)):
            return

        results = self.model(rgb)
        overlay_info = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                u, v = map(int, box.xywh[0][:2].cpu().numpy())
                if not (0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]):
                    continue

                z = float(depth[v, u]) / 1000.0
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.classNames[cls] if cls < len(self.classNames) else f'class_{cls}'

                pt_camera = PointStamped()
                pt_camera.header.frame_id = rgb_msg.header.frame_id
                pt_camera.header.stamp = rclpy.time.Time().to_msg()
                pt_camera.point.x, pt_camera.point.y, pt_camera.point.z = x, y, z

                map_x, map_y, map_z = self.transform_to_map(pt_camera, label)

                overlay_info.append({
                    "label": label,
                    "conf": conf,
                    "center": (u, v),
                    "bbox": (x1, y1, x2, y2),
                    "depth": z
                })

        with self.lock:
            self.overlay_info = overlay_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model file (.pt, .onnx, .engine)')
    args, _ = parser.parse_known_args()

    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    model = YOLO(model_path)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using GPU for inference." if torch.cuda.is_available() else "Using CPU.")

    rclpy.init()
    node = YoloDepthToMap(model)

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        while rclpy.ok() and not node.should_shutdown:
            with node.lock:
                frame = node.latest_rgb.copy() if node.latest_rgb is not None else None
                overlay_info = node.overlay_info.copy()

            if frame is not None:
                for obj in overlay_info:
                    u, v = obj["center"]
                    x1, y1, x2, y2 = obj["bbox"]
                    label = obj["label"]
                    conf = obj["conf"]
                    z = obj["depth"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (u, v), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"{label} {conf:.2f} {z:.2f}m", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                cv2.imshow("YOLO + Depth + Map", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                node.get_logger().info("Shutdown requested by user.")
                node.should_shutdown = True
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        print("Shutdown complete.")
        sys.exit(0)

if __name__ == '__main__':
    main()
