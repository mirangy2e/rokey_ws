# depth_checker_with_camerainfo.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
import cv2
from cv_bridge import CvBridge

# ================================
# 설정 상수
# ================================
DEPTH_TOPIC = '/oakd/rgb/preview/depth'  # Depth 이미지 토픽
CAMERA_INFO_TOPIC = '/oakd/rgb/preview/camera_info'  # CameraInfo 토픽
MAX_DEPTH_METERS = 20.0                 # Depth 시각화 최대 거리(m)
NORMALIZE_DEPTH_RANGE = 20.0             # 정규화할 Depth 범위(m)
# ================================

class DepthChecker(Node):
    def __init__(self):
        super().__init__('depth_checker')
        self.bridge = CvBridge()

        # 카메라 내부 파라미터 (K 매트릭스)
        self.K = None

        self.should_exit = False  # 종료 요청 플래그

        # Depth 구독
        self.subscription = self.create_subscription(
            Image,
            DEPTH_TOPIC,
            self.depth_callback,
            10)

        # CameraInfo 구독
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            CAMERA_INFO_TOPIC,
            self.camera_info_callback,
            10)

    def camera_info_callback(self, msg):
        if self.K is None:  # 최초 1회만 저장
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(f"CameraInfo received: fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}")

    def depth_callback(self, msg):
        if self.should_exit:
            return

        if self.K is None:
            self.get_logger().warn('Waiting for CameraInfo...')
            return

        # Depth 이미지를 float32로 변환
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        height, width = depth_image.shape

        # CameraInfo로부터 중심점 가져오기
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        u, v = int(cx), int(cy)

        # 중심 픽셀의 거리값
        distance = depth_image[v, u]
        self.get_logger().info(f"Image size: {width}x{height}, Distance at (u={u}, v={v}) = {distance:.2f} meters")

        # 시각화를 위해 depth를 8bit로 정규화
        depth_vis = np.nan_to_num(depth_image, nan=0.0)
        depth_vis = np.clip(depth_vis, 0, MAX_DEPTH_METERS)
        depth_vis = (depth_vis / NORMALIZE_DEPTH_RANGE * 255).astype(np.uint8)

        # 컬러맵 적용
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # 중심점 그리기
        cv2.circle(depth_colored, (u, v), 5, (0, 0, 0), -1)  # 검정색 점
        cv2.line(depth_colored, (0, v), (width, v), (0, 0, 0), 1)  # 가로선
        cv2.line(depth_colored, (u, 0), (u, height), (0, 0, 0), 1)  # 세로선

        # 화면에 표시
        cv2.imshow('Depth Image with Center Mark', depth_colored)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.should_exit = True  # 종료 요청만 설정

def main():
    rclpy.init()
    node = DepthChecker()

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
