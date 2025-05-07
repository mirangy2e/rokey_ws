import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2

# ============================
# 설정 상수
# ============================
ROBOT_NAMESPACE = 'robot4'
RGB_INPUT_TOPIC = f'/{ROBOT_NAMESPACE}/oakd/rgb/preview/image_raw'
DEPTH_INPUT_TOPIC = f'/{ROBOT_NAMESPACE}/oakd/stereo/image_raw'
CAMERA_INFO_INPUT_TOPIC = f'/{ROBOT_NAMESPACE}/oakd/stereo/camera_info'

RGB_OUTPUT_TOPIC = f'/{ROBOT_NAMESPACE}/cropped/rgb/image_raw'
DEPTH_OUTPUT_TOPIC = f'/{ROBOT_NAMESPACE}/cropped/depth/image_raw'
CAMERA_INFO_OUTPUT_TOPIC = f'/{ROBOT_NAMESPACE}/cropped/camera_info'

CROP_WIDTH = 480
CROP_HEIGHT = 256
ORIGINAL_WIDTH = 480
ORIGINAL_HEIGHT = 480
# ============================


def crop_bottom_center(image, crop_width, crop_height):
    h, w = image.shape[:2]
    center_x = w // 2
    center_y = h

    x_start = center_x - crop_width // 2
    y_start = center_y - crop_height

    cropped = image[y_start:y_start+crop_height, x_start:x_start+crop_width]
    return cropped, x_start, y_start

def adjust_camera_info_for_crop(raw_info, x_offset, y_offset, crop_width, crop_height):
    cropped_info = CameraInfo()
    cropped_info.header = raw_info.header

    cropped_info.width = crop_width
    cropped_info.height = crop_height
    cropped_info.distortion_model = raw_info.distortion_model
    cropped_info.d = list(raw_info.d)
    cropped_info.r = list(raw_info.r)
    cropped_info.binning_x = raw_info.binning_x
    cropped_info.binning_y = raw_info.binning_y
    cropped_info.roi = raw_info.roi

    # K matrix 수정
    K = list(raw_info.k)
    K[2] -= x_offset  # cx
    K[5] -= y_offset  # cy
    cropped_info.k = K

    # P matrix 수정
    P = list(raw_info.p)
    P[2] -= x_offset  # cx
    P[6] -= y_offset  # cy
    cropped_info.p = P

    return cropped_info

class CroppedImagePublisher(Node):
    def __init__(self):
        super().__init__('cropped_image_publisher')
        self.bridge = CvBridge()

        # 퍼블리셔
        self.rgb_pub = self.create_publisher(Image, RGB_OUTPUT_TOPIC, 10)
        self.depth_pub = self.create_publisher(Image, DEPTH_OUTPUT_TOPIC, 10)
        self.caminfo_pub = self.create_publisher(CameraInfo, CAMERA_INFO_OUTPUT_TOPIC, 10)

        # 구독자
        self.create_subscription(Image, RGB_INPUT_TOPIC, self.rgb_callback, 10)
        self.create_subscription(Image, DEPTH_INPUT_TOPIC, self.depth_callback, 10)
        self.create_subscription(CameraInfo, CAMERA_INFO_INPUT_TOPIC, self.camera_info_callback, 10)

        self.raw_camera_info = None  # 원본 camera_info 저장

    def camera_info_callback(self, msg):
        self.raw_camera_info = msg  # camera_info 저장

    def rgb_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cropped, x_offset, y_offset = crop_bottom_center(image, CROP_WIDTH, CROP_HEIGHT)
            msg_out = self.bridge.cv2_to_imgmsg(cropped, encoding='bgr8')
            msg_out.header = msg.header
            self.rgb_pub.publish(msg_out)

            # CameraInfo도 같이 publish
            if self.raw_camera_info:
                adjusted_info = adjust_camera_info_for_crop(self.raw_camera_info, x_offset, y_offset, CROP_WIDTH, CROP_HEIGHT)
                adjusted_info.header = msg.header  # 타임스탬프 맞추기
                self.caminfo_pub.publish(adjusted_info)

        except Exception as e:
            self.get_logger().error(f"RGB crop error: {e}")

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            cropped, _, _ = crop_bottom_center(depth, CROP_WIDTH, CROP_HEIGHT)
            msg_out = self.bridge.cv2_to_imgmsg(cropped, encoding='passthrough')
            msg_out.header = msg.header
            self.depth_pub.publish(msg_out)
        except Exception as e:
            self.get_logger().error(f"Depth crop error: {e}")

def main():
    rclpy.init()
    node = CroppedImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
