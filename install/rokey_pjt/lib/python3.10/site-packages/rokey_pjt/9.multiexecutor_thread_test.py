import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import threading
import cv2
import time
import numpy as np

class ImageGeneratorNode(Node):
    def __init__(self):
        super().__init__('image_generator_node')
        self.get_logger().info("Node started.")
        self.frame = None
        self.lock = threading.Lock()

        # 30 FPS로 dummy 이미지 생성
        self.create_timer(1.0 / 30.0, self.timer_callback)

    def timer_callback(self):
        # OpenCV로 흰색 바탕에 현재 시간 텍스트 표시
        img = 255 * np.ones((480, 640, 3), dtype=np.uint8)
        timestamp = self.get_clock().now().to_msg()
        text = f"Time: {timestamp.sec}.{timestamp.nanosec//1000000:03d}"
        cv2.putText(img, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # 공유 변수에 저장
        with self.lock:
            self.frame = img

def main():
    rclpy.init()
    node = ImageGeneratorNode()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    # executor를 별도 스레드에서 실행
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        while rclpy.ok():
            with node.lock:
                if node.frame is not None:
                    cv2.imshow("ROS2 + OpenCV Demo", node.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
