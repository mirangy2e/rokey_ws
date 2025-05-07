# yolo_rgb_checker.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

# ================================
# 설정 상수
# ================================
YOLO_MODEL_PATH = '/home/mi/rokey_ws/model/yolov8n.pt'  # YOLO 모델 경로
RGB_TOPIC = '/oakd/rgb/preview/image_raw'      # RGB 이미지 토픽 (Gazebo용)
# ================================

class YoloRgbChecker(Node):
    def __init__(self):
        super().__init__('yolo_rgb_checker')
        self.bridge = CvBridge()

        # YOLO 모델 로드
        self.model = YOLO(YOLO_MODEL_PATH)

        # 종료 요청 플래그
        self.should_exit = False

        # RGB 이미지 구독
        self.rgb_sub = self.create_subscription(
            Image,
            RGB_TOPIC,
            self.rgb_callback,
            10)

    def rgb_callback(self, msg):
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
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0])
                label_idx = int(box.cls[0])
                label = self.model.names[label_idx]

                # 박스 그리기
                cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(rgb_image, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 결과 화면에 표시
        cv2.imshow('YOLO Detection', rgb_image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.should_exit = True  # 종료 요청

def main():
    rclpy.init()
    node = YoloRgbChecker()

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
