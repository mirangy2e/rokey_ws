import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, PoseStamped
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
import torch

# === 상수 정의 ===
MODEL_PATH = '/home/mi/rokey_ws/model/my_best.pt'
TARGET_CLASS_ID = 0  # 예: car 클래스
INFERENCE_PERIOD_SEC = 1.0 / 20  # YOLO 추론 주기 20Hz

GOAL_POSE_TOPIC = '/yolo_goal_pose'
GOAL_PUBLISH_INTERVAL_SEC = 2.0  # goal_pose 퍼블리시 최소 주기 (초)

RGB_TOPIC = 'cropped/rgb/image_raw'
DEPTH_TOPIC = 'cropped/depth/image_raw'
CAMERA_INFO_TOPIC = 'cropped/camera_info'

class YoloDepthToMap(Node):
    def __init__(self):
        super().__init__('yolo_depth_to_map')
        self.get_logger().info("[1/6] 노드 초기화 시작...")

        if not os.path.exists(MODEL_PATH):
            self.get_logger().error(f"Model file not found: {MODEL_PATH}")
            sys.exit(1)

        self.model = YOLO(MODEL_PATH)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"[2/6] YOLO 모델 로드 완료 (GPU 사용: {torch.cuda.is_available()})")

        self.bridge = CvBridge()
        self.classNames = getattr(self.model, 'names', [])

        self.K = None
        self.latest_rgb = self.latest_depth = self.latest_rgb_msg = None
        self.overlay_info = []
        self.display_rgb = None
        self.lock = threading.Lock()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.get_logger().info("[3/6] TF2 Transform Listener 초기화 완료")

        self.create_subscription(Image, RGB_TOPIC, self.rgb_callback, 1)
        self.create_subscription(Image, DEPTH_TOPIC, self.depth_callback, 1)
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, 1)
        self.get_logger().info(f"[4/6] RGB, Depth, CameraInfo 구독자 생성 완료")

        self.pose_pub = self.create_publisher(PoseStamped, GOAL_POSE_TOPIC, 10)
        self.last_goal_time = self.get_clock().now()
        self.get_logger().info(f"[5/6] Detected Map 토픽 생성 완료")

        self.create_timer(INFERENCE_PERIOD_SEC, self.inference_callback)
        self.get_logger().info(f"[6/6] 추론 루프 설정 완료: {1.0 / INFERENCE_PERIOD_SEC:.1f} Hz")

    def can_publish_goal(self):
        """퍼블리시 간격 초과 여부 판단"""
        now = self.get_clock().now()
        elapsed = (now - self.last_goal_time).nanoseconds * 1e-9
        if elapsed >= GOAL_PUBLISH_INTERVAL_SEC: 
            self.last_goal_time = now
            return True
        return False

    def camera_info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(f"CameraInfo 수신: fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}")

    def rgb_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self.lock:
                self.latest_rgb = img.copy()
                self.latest_rgb_msg = msg
                self.display_rgb = img.copy()
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

        results = self.model(rgb, stream=True)
        overlay_info = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls != TARGET_CLASS_ID:
                    continue

                u, v = map(int, box.xywh[0][:2].cpu().numpy())
                if not (0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]):
                    continue

                z = float(depth[v, u]) / 1000.0
                if z <= 0.05 or np.isnan(z):
                    continue

                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                label = self.classNames[cls] if cls < len(self.classNames) else f'class_{cls}'

                pt_camera = PointStamped()
                pt_camera.header.frame_id = rgb_msg.header.frame_id
                pt_camera.header.stamp = rclpy.time.Time().to_msg()
                pt_camera.point.x, pt_camera.point.y, pt_camera.point.z = x, y, z

                map_x, map_y, map_z = self.transform_to_map(pt_camera, label)
                if not np.isnan(map_x) and self.can_publish_goal():
                    goal = PoseStamped()
                    goal.header.frame_id = 'map'
                    goal.header.stamp = self.get_clock().now().to_msg()
                    goal.pose.position.x = map_x
                    goal.pose.position.y = map_y
                    goal.pose.position.z = 0.0
                    goal.pose.orientation.w = 1.0
                    self.pose_pub.publish(goal)
                    self.get_logger().info(f"[GOAL] goal_pose published: x={map_x:.2f}, y={map_y:.2f}")

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
    rclpy.init()
    node = YoloDepthToMap()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        while rclpy.ok():
            with node.lock:
                frame = node.display_rgb.copy() if node.display_rgb is not None else None
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
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == '__main__':
    main()
