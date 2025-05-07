import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, CameraInfo
import threading
import cv2
import time
import numpy as np

class DummyCameraNode(Node):
    def __init__(self):
        super().__init__('dummy_camera_node')

        self.latest_rgb = None
        self.latest_depth = None
        self.K = None
        self.lock = threading.Lock()

        self.rgb_counter = 0
        self.depth_counter = 0
        self.camera_info_counter = 0

        self.create_timer(1.0 / 30.0, self.rgb_callback)         # 30Hz
        self.create_timer(1.0 / 15.0, self.depth_callback)       # 15Hz
        self.create_timer(1.0 / 1.0, self.camera_info_callback)  # 1Hz

    def rgb_callback(self):
        self.rgb_counter += 1
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, f"RGB Frame {self.rgb_counter}", (160, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        with self.lock:
            self.latest_rgb = img

        self.get_logger().info(f"[RGB] Frame #{self.rgb_counter}")

    def depth_callback(self):
        self.depth_counter += 1
        img = np.zeros((480, 640), dtype=np.uint8)
        color = 50 + (self.depth_counter * 10) % 205
        img[:] = color
        cv2.putText(img, f"DEPTH {self.depth_counter}", (180, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
        img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        with self.lock:
            self.latest_depth = img_colored

        self.get_logger().info(f"[DEPTH] Frame #{self.depth_counter}")

    def camera_info_callback(self):
        self.camera_info_counter += 1
        fx = 500 + (self.camera_info_counter % 5) * 5
        K = np.array([[fx, 0.0, 319.5],
                      [0.0, fx, 239.5],
                      [0.0, 0.0, 1.0]])
        with self.lock:
            self.K = K

        self.get_logger().info(f"[CameraInfo] fx={fx}")

def main():
    rclpy.init()
    node = DummyCameraNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        while rclpy.ok():
            combined = np.zeros((480, 640, 3), dtype=np.uint8)
            with node.lock:
                if node.latest_rgb is not None:
                    combined = node.latest_rgb.copy()
                if node.latest_depth is not None:
                    d_img = cv2.resize(node.latest_depth, (160, 120))
                    combined[0:120, -160:] = d_img
                if node.K is not None:
                    fx = node.K[0, 0]
                    cv2.putText(combined, f"fx={fx:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("RGB + Depth + Info (Mock)", combined)
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
