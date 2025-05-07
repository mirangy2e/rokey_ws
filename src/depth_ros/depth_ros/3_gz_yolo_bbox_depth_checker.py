# yolo_bbox_depth_checker_with_camerainfo_safe_v2.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2
from ultralytics import YOLO

# ================================
# 설정 상수
# ================================
YOLO_MODEL_PATH = '/home/mi/rokey_ws/model/yolov8n.pt'  # YOLO 모델 경로
RGB_TOPIC = '/oakd/rgb/preview/image_raw'      # RGB 이미지 토픽
DEPTH_TOPIC = '/oakd/rgb/preview/depth'         # Depth 이미지 토픽
CAMERA_INFO_TOPIC = '/oakd/rgb/preview/camera_info'  # CameraInfo 토픽
# ================================

class YoloBboxDepthChecker(Node):
    def __init__(self):
        super().__init__('yolo_bbox_depth_checker')
        self.bridge = CvBridge()

        # YOLO 모델 로드
        self.model = YOLO(YOLO_MODEL_PATH)

        # Depth 이미지 저장
        self.latest_depth_image = None

        # 카메라 내부 파라미터
        self.K = None  # 3x3 intrinsic matrix

        # 종료 요청 플래그
        self.should_exit = False

        # Depth 구독
        self.depth_sub = self.create_subscription(
            Image,
            DEPTH_TOPIC,
            self.depth_callback,
            10)

        # RGB 구독
        self.rgb_sub = self.create_subscription(
            Image,
            RGB_TOPIC,
            self.rgb_callback,
            10)

        # CameraInfo 구독
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            CAMERA_INFO_TOPIC,
            self.camera_info_callback,
            10)

    def camera_info_callback(self, msg):
        if self.K is None:  # 이미 세팅되어 있으면 무시
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(f"CameraInfo received: fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}")

    def depth_callback(self, msg):
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {e}")

    def rgb_callback(self, msg):
        if self.latest_depth_image is None or self.K is None:
            self.get_logger().warn("Waiting for depth image or camera intrinsics...")
            return

        try:
            rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"RGB image conversion failed: {e}")
            return

        # YOLO로 객체 검출
        results = self.model(rgb_image)

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x_center, y_center, width, height = box.xywh[0].cpu().numpy()
                u, v = int(x_center), int(y_center)

                # Depth 이미지 범위 확인
                if (0 <= v < self.latest_depth_image.shape[0]) and (0 <= u < self.latest_depth_image.shape[1]):
                    z = float(self.latest_depth_image[v, u])

                    # (u,v,z) → (x,y,z) 변환
                    fx = self.K[0, 0]
                    fy = self.K[1, 1]
                    cx = self.K[0, 2]
                    cy = self.K[1, 2]

                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy

                    self.get_logger().info(f"Object center at ({u},{v}) → Depth: {z:.2f}m, Camera coords: ({x:.2f}, {y:.2f}, {z:.2f})")

                    # 시각화용: bbox 그리기
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(rgb_image, (u, v), 5, (0, 0, 255), -1)
                    cv2.putText(rgb_image, f"{z:.2f}m", (u, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                else:
                    self.get_logger().warn(f"Center ({u},{v}) out of depth image bounds.")

        # 결과 화면에 표시
        cv2.imshow('YOLO + Depth + CameraInfo', rgb_image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.should_exit = True  # 종료 요청만

def main():
    rclpy.init()
    node = YoloBboxDepthChecker()

    try:
        while rclpy.ok() and not node.should_exit:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
