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

from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator, TurtleBot4Directions

# =============================
# 설정 상수
# =============================
YOLO_MODEL_PATH = '/home/mi/rokey_ws/model/yolov8n.pt'
RGB_TOPIC = '/oakd/rgb/preview/image_raw'
DEPTH_TOPIC = '/oakd/rgb/preview/depth'
CAMERA_INFO_TOPIC = '/oakd/rgb/preview/camera_info'

GOAL_DISTANCE_THRESHOLD = 2.0
GOAL_REACHED_THRESHOLD = 2.0
# =============================


class YoloNavigator(Node):
    def __init__(self):
        super().__init__('yolo_nav_person_with_display')
        self.bridge = CvBridge()
        self.model = YOLO(YOLO_MODEL_PATH)

        self.latest_rgb = None
        self.latest_depth = None
        self.latest_rgb_msg = None
        self.K = None
        self.last_goal = None
        self.goal_sent = False

        self.lock = threading.Lock()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.navigator = TurtleBot4Navigator()

        self.create_subscription(Image, RGB_TOPIC, self.rgb_callback, 10)
        self.create_subscription(Image, DEPTH_TOPIC, self.depth_callback, 10)
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, 10)

        self.thread = threading.Thread(target=self.run_detection_loop)
        self.thread.start()

    def camera_info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info("CameraInfo received.")

    def rgb_callback(self, msg):
        try:
            with self.lock:
                self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.latest_rgb_msg = msg
        except Exception as e:
            self.get_logger().error(f"RGB error: {e}")

    def depth_callback(self, msg):
        try:
            with self.lock:
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth error: {e}")

    def is_significant_movement(self, new_goal):
        if self.last_goal is None:
            self.last_goal = new_goal
            return True
        dx = new_goal[0] - self.last_goal[0]
        dy = new_goal[1] - self.last_goal[1]
        if np.hypot(dx, dy) > GOAL_DISTANCE_THRESHOLD:
            self.last_goal = new_goal
            return True
        return False

    def run_detection_loop(self):
        while rclpy.ok():
            with self.lock:
                if self.latest_rgb is None or self.latest_depth is None or self.K is None:
                    continue
                rgb = self.latest_rgb.copy()
                depth = self.latest_depth.copy()
                rgb_msg = self.latest_rgb_msg

            if self.goal_sent:
                feedback = self.navigator.getFeedback()
                if feedback and feedback.distance_remaining < GOAL_REACHED_THRESHOLD:
                    self.get_logger().info("[목표 도달] 새로운 탐지를 기다립니다.")
                    self.goal_sent = False
                else:
                    continue

            results = self.model(rgb)
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]

                    if class_name != "person":
                        continue

                    x_center, y_center, _, _ = box.xywh[0].cpu().numpy()
                    u, v = int(x_center), int(y_center)

                    if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
                        continue
                    z = float(depth[v, u])
                    if z == 0.0:
                        continue

                    fx, fy = self.K[0, 0], self.K[1, 1]
                    cx, cy = self.K[0, 2], self.K[1, 2]
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy

                    # 시각화: bbox, 중심점, depth 텍스트
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(rgb, (u, v), 5, (0, 0, 255), -1)
                    cv2.putText(rgb, f"{z:.2f}m", (u, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    pt_cam = PointStamped()
                    pt_cam.header.stamp = rclpy.time.Time().to_msg()
                    pt_cam.header.frame_id = rgb_msg.header.frame_id
                    pt_cam.point.x = x
                    pt_cam.point.y = y
                    pt_cam.point.z = z

                    try:
                        pt_map = self.tf_buffer.transform(pt_cam, 'map', rclpy.duration.Duration(seconds=0.5))
                        goal_xy = [pt_map.point.x, pt_map.point.y]

                        if not self.is_significant_movement(goal_xy):
                            self.get_logger().info(
                                "\n[유사 목표로 이동 생략됨]\n"
                                f"- 현재 목표 좌표: x={self.last_goal[0]:.2f}, y={self.last_goal[1]:.2f}\n"
                                f"- 새로운 좌표   : x={goal_xy[0]:.2f}, y={goal_xy[1]:.2f}"
                            )
                            continue

                        self.get_logger().info(
                            "\n[새로운 사람 탐지 → 이동 시작]\n"
                            f"- 목표 좌표: x={goal_xy[0]:.2f}, y={goal_xy[1]:.2f}"
                        )

                        self.navigator.cancelTask()
                        goal_pose = self.navigator.getPoseStamped(goal_xy, TurtleBot4Directions.EAST)
                        self.navigator.startToPose(goal_pose)
                        self.goal_sent = True

                    except Exception as e:
                        self.get_logger().warn(f"TF transform failed: {e}")
                    break

            # 결과 화면 출력
            cv2.imshow('YOLO + Depth + Navigation', rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def main():
    rclpy.init()
    node = YoloNavigator()
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
