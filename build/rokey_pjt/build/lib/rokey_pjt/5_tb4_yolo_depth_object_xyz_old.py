import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import numpy as np
import cv2
import tf2_ros
import tf2_geometry_msgs
from ultralytics import YOLO
import threading
import torch
import os

# ========================
# 상수 정의
# ========================
YOLO_MODEL_PATH = '/home/mi/rokey_ws/model/my_best.pt'

RGB_TOPIC = 'cropped/rgb/image_raw'
DEPTH_TOPIC = 'cropped/depth/image_raw'
CAMERA_INFO_TOPIC = 'cropped/camera_info'

TARGET_CLASS_ID = 0  # YOLO 클래스 ID: 0 = Car
MAX_DEPTH_METERS = 20.0
DEPTH_PATCH_SIZE = 7
# ========================

class YoloDepthToMap(Node):
    def __init__(self):
        super().__init__('yolo_depth_to_map')
        self.get_logger().info("YoloDepthToMap node is starting...")
        self.bridge = CvBridge()
        self.model = YOLO(YOLO_MODEL_PATH)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

        self.latest_depth_image = None
        self.latest_rgb_image = None
        self.latest_rgb_msg = None
        self.K = None
        self.processed_frame = None

        self.lock = threading.Lock()
        self.should_exit = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(Image, DEPTH_TOPIC, self.depth_callback, 10)
        self.create_subscription(Image, RGB_TOPIC, self.rgb_callback, 10)
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, 10)

    def camera_info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(f"CameraInfo: fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}")

    def depth_callback(self, msg):
        try:
            with self.lock:
                self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {e}")

    def rgb_callback(self, msg):
        try:
            with self.lock:
                self.latest_rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                self.latest_rgb_msg = msg
        except Exception as e:
            self.get_logger().error(f"RGB image conversion failed: {e}")

    def get_stable_depth(self, u, v, depth_image, patch_size=DEPTH_PATCH_SIZE, max_depth=MAX_DEPTH_METERS):
        half = patch_size // 2
        u_min = max(u - half, 0)
        u_max = min(u + half + 1, depth_image.shape[1])
        v_min = max(v - half, 0)
        v_max = min(v + half + 1, depth_image.shape[0])
        roi = depth_image[v_min:v_max, u_min:u_max]
        valid = roi[roi > 0]
        if valid.size == 0:
            return None, u_min, u_max, v_min, v_max
        z = float(np.median(valid))
        if z <= 0.0 or z > max_depth * 1000:
            return None, u_min, u_max, v_min, v_max
        return z, u_min, u_max, v_min, v_max

    def inference_loop(self):
        while not self.should_exit and rclpy.ok():
            with self.lock:
                if self.latest_rgb_image is None or self.latest_depth_image is None or self.K is None:
                    continue
                rgb_image = self.latest_rgb_image.copy()
                depth_image = self.latest_depth_image.copy()
                rgb_msg = self.latest_rgb_msg

            depth_blurred = cv2.GaussianBlur(depth_image, (5, 5), 0)
            results = self.model(rgb_image)

            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    if class_id != TARGET_CLASS_ID:
                        continue

                    x_center, y_center, w, h = box.xywh[0].cpu().numpy()
                    u, v = int(x_center), int(y_center)
                    if not (0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]):
                        continue

                    z, u_min, u_max, v_min, v_max = self.get_stable_depth(u, v, depth_blurred)
                    if z is None:
                        self.get_logger().info(f"Invalid depth at ({u},{v}), skipping")
                        continue
                    z /= 1000.0  # mm → meters

                    fx, fy = self.K[0, 0], self.K[1, 1]
                    cx, cy = self.K[0, 2], self.K[1, 2]
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy

                    pt_camera = PointStamped()
                    pt_camera.header = rgb_msg.header
                    pt_camera.point.x = x
                    pt_camera.point.y = y
                    pt_camera.point.z = z

                    try:
                        pt_map = self.tf_buffer.transform(pt_camera, 'map', timeout=rclpy.duration.Duration(seconds=0.5))
                        self.get_logger().info(
                            f"[YOLO DETECTED] map: x={pt_map.point.x:.2f}, y={pt_map.point.y:.2f}, z={pt_map.point.z:.2f}"
                        )
                    except Exception as e:
                        self.get_logger().warn(f"TF 변환 실패: {e}")

                    # 시각화용 이미지 저장
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(rgb_image, (u, v), 5, (0, 0, 255), -1)
                    cv2.putText(rgb_image, f"{z:.2f}m", (u, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.rectangle(rgb_image, (u_min, v_min), (u_max - 1, v_max - 1), (0, 0, 255), 1)

                    break  # 첫 번째 유효 객체만 처리

            with self.lock:
                self.processed_frame = rgb_image.copy()

    def destroy_node(self):
        self.should_exit = True
        super().destroy_node()

def main():
    rclpy.init()
    node = YoloDepthToMap()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    # 추론 루프 시작
    threading.Thread(target=node.inference_loop, daemon=True).start()

    try:
        # OpenCV GUI 루프
        while rclpy.ok() and not node.should_exit:
            with node.lock:
                frame = node.processed_frame
            if frame is not None:
                cv2.imshow('YOLO + Depth + Map', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                node.should_exit = True
                break
            rclpy.spin_once(node, timeout_sec=0.001)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
