#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import numpy as np
import cv2
import tf2_ros
import tf2_geometry_msgs
from ultralytics import YOLO
import threading

# ================================
# 설정 상수
# ================================
YOLO_MODEL_PATH = '/home/mi/rokey_ws/model/yolov8n.pt'
RGB_TOPIC = '/oakd/rgb/preview/image_raw'
DEPTH_TOPIC = '/oakd/rgb/preview/depth'
CAMERA_INFO_TOPIC = '/oakd/rgb/preview/camera_info'
# ================================

class YoloDepthToMap(Node):
    def __init__(self):
        super().__init__('yolo_depth_to_map')

        self.bridge = CvBridge()
        self.model = YOLO(YOLO_MODEL_PATH)  # YOLO 모델 로드

        self.latest_depth_image = None
        self.latest_rgb_image = None
        self.latest_rgb_msg = None
        self.K = None  # 카메라 내부 파라미터 행렬

        self.lock = threading.Lock()
        self.should_exit = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 이미지 및 카메라 정보 구독
        self.depth_sub = self.create_subscription(Image, DEPTH_TOPIC, self.depth_callback, 10)
        self.rgb_sub = self.create_subscription(Image, RGB_TOPIC, self.rgb_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, 10)

        # 객체 인식 및 거리 측정을 위한 처리 루프 실행
        self.processing_thread = threading.Thread(target=self.processing_loop)
        self.processing_thread.start()

    def camera_info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(f"CameraInfo received: fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}")

    def depth_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.lock:
                self.latest_depth_image = depth_image
        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {e}")

    def rgb_callback(self, msg):
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.latest_rgb_image = rgb_image
                self.latest_rgb_msg = msg
        except Exception as e:
            self.get_logger().error(f"RGB image conversion failed: {e}")

    def get_stable_depth(self, u, v, depth_image, patch_size=5, max_depth=10.0):
        """
        중심점 주변 패치에서 중앙값 기반으로 안정적인 depth 값을 반환한다.
        노이즈 완화 및 신뢰도 향상을 위해 사용됨.
        """
        half = patch_size // 2
        u_min = max(u - half, 0)
        u_max = min(u + half + 1, depth_image.shape[1])
        v_min = max(v - half, 0)
        v_max = min(v + half + 1, depth_image.shape[0])

        roi = depth_image[v_min:v_max, u_min:u_max]
        valid_pixels = roi[roi > 0]

        if valid_pixels.size == 0:
            return None, None, None, None

        z = float(np.median(valid_pixels))

        # 깊이값 유효성 검사
        if z <= 0.0 or z > max_depth:
            return None, None, None, None

        return z, u_min, u_max, v_min, v_max

    def processing_loop(self):
        """
        YOLO 객체 인식 + depth 처리 루프
        - 객체 중심의 거리 계산
        - 좌표 변환 (카메라 → map)
        - 시각화 처리
        """
        while not self.should_exit:
            with self.lock:
                if self.latest_rgb_image is None or self.latest_depth_image is None or self.K is None:
                    continue
                rgb_image = self.latest_rgb_image.copy()
                depth_image = self.latest_depth_image.copy()
                rgb_msg = self.latest_rgb_msg

            # Gaussian blur 적용하여 depth 노이즈 완화
            depth_blurred = cv2.GaussianBlur(depth_image, (5, 5), 0)

            results = self.model(rgb_image)

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    x_center, y_center, width, height = box.xywh[0].cpu().numpy()
                    u, v = int(x_center), int(y_center)  # 중심점 (픽셀 좌표)

                    if not (0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]):
                        self.get_logger().warn(f"Center ({u},{v}) out of depth image bounds.")
                        continue

                    # 안정적인 depth 값 계산
                    z, u_min, u_max, v_min, v_max = self.get_stable_depth(u, v, depth_blurred)
                    if z is None:
                        continue

                    # 카메라 좌표계로 변환
                    fx, fy = self.K[0, 0], self.K[1, 1]
                    cx, cy = self.K[0, 2], self.K[1, 2]

                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy

                    self.get_logger().info(f"[Camera Frame] ({u},{v}) → (x={x:.2f}, y={y:.2f}, z={z:.2f})")

                    # 좌표 변환: camera → map
                    pt_camera = PointStamped()
                    pt_camera.header.frame_id = rgb_msg.header.frame_id
                    pt_camera.header.stamp = rgb_msg.header.stamp
                    pt_camera.point.x = x
                    pt_camera.point.y = y
                    pt_camera.point.z = z

                    try:
                        pt_map = self.tf_buffer.transform(
                            pt_camera,
                            'map',
                            timeout=rclpy.duration.Duration(seconds=0.5)
                        )
                        self.get_logger().info(f"[Map Frame] Object at map → (x={pt_map.point.x:.2f}, y={pt_map.point.y:.2f}, z={pt_map.point.z:.2f})")
                    except Exception as e:
                        self.get_logger().warn(f"TF transform to map failed: {e}")

                    # ========================
                    # 시각화: bbox, 중심점, 거리, 패치
                    # ========================
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(rgb_image, (u, v), 5, (0, 0, 255), -1)
                    cv2.putText(rgb_image, f"{z:.2f}m", (u, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # 패치 영역 시각화 (얇은 빨간 사각형)
                    cv2.rectangle(rgb_image, (u_min, v_min), (u_max - 1, v_max - 1), (0, 0, 255), 1)

            # 결과 화면 출력
            cv2.imshow('YOLO + Depth + Map', rgb_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.should_exit = True
                break

    def destroy_node(self):
        self.should_exit = True
        super().destroy_node()

def main():
    rclpy.init()
    node = YoloDepthToMap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
