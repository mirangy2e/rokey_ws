import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, PoseStamped
from cv_bridge import CvBridge
import numpy as np
import cv2
import tf2_ros
import tf2_geometry_msgs
from ultralytics import YOLO
import threading

YOLO_MODEL_PATH = '/home/mi/rokey_ws/model/yolov8n.pt'
RGB_TOPIC = '/oakd/rgb/preview/image_raw'
DEPTH_TOPIC = '/oakd/rgb/preview/depth'
CAMERA_INFO_TOPIC = '/oakd/rgb/preview/camera_info'
GOAL_POSE_TOPIC = '/yolo_goal_pose'

class YoloDepthPublisher(Node):
    def __init__(self):
        super().__init__('yolo_depth_publisher')
        self.bridge = CvBridge()
        self.model = YOLO(YOLO_MODEL_PATH)

        self.latest_depth_image = None
        self.latest_rgb_image = None
        self.latest_rgb_msg = None
        self.K = None

        self.lock = threading.Lock()
        self.should_exit = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ROS topic 구독
        self.create_subscription(Image, DEPTH_TOPIC, self.depth_callback, 10)
        self.create_subscription(Image, RGB_TOPIC, self.rgb_callback, 10)
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, 10)

        # 목표 좌표 publisher
        self.pose_pub = self.create_publisher(PoseStamped, GOAL_POSE_TOPIC, 10)

        # YOLO 처리용 thread 시작
        threading.Thread(target=self.processing_loop, daemon=True).start()

    def camera_info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(f"CameraInfo received: fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}")

    def depth_callback(self, msg):
        try:
            with self.lock:
                self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {e}")

    def rgb_callback(self, msg):
        try:
            with self.lock:
                self.latest_rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.latest_rgb_msg = msg
        except Exception as e:
            self.get_logger().error(f"RGB image conversion failed: {e}")

    def processing_loop(self):
        cv2.namedWindow('YOLO + Depth + Map', cv2.WINDOW_NORMAL)
        while not self.should_exit:
            with self.lock:
                if self.latest_rgb_image is None or self.latest_depth_image is None or self.K is None:
                    continue

                rgb_image = self.latest_rgb_image.copy()
                depth_image = self.latest_depth_image.copy()
                rgb_msg = self.latest_rgb_msg

            results = self.model(rgb_image)
            for result in results:
                for box in result.boxes:
                    x_center, y_center, w, h = box.xywh[0].cpu().numpy()
                    u, v = int(x_center), int(y_center)

                    if not (0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]):
                        continue

                    z = float(depth_image[v, u])
                    if z == 0.0:
                        continue

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
                        self.get_logger().info(f" Publish goal: map (x={pt_map.point.x:.2f}, y={pt_map.point.y:.2f}, z={pt_map.point.z:.2f})")

                        goal_pose = PoseStamped()
                        goal_pose.header.frame_id = 'map'
                        goal_pose.header.stamp = rclpy.time.Time().to_msg()
                        goal_pose.pose.position.x = pt_map.point.x
                        goal_pose.pose.position.y = pt_map.point.y
                        goal_pose.pose.position.z = 0.0
                        goal_pose.pose.orientation.w = 1.0  # 단순 정방향

                        self.pose_pub.publish(goal_pose)

                        # 시각화
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(rgb_image, (u, v), 5, (0, 0, 255), -1)
                        cv2.putText(rgb_image, f"{z:.2f}m", (u, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        break  # 하나만 publish
                    except Exception as e:
                        self.get_logger().warn(f"TF 변환 실패: {e}")

            cv2.imshow('YOLO + Depth + Map', rgb_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.should_exit = True
                break

    def destroy_node(self):
        self.should_exit = True
        super().destroy_node()

def main():
    rclpy.init()
    node = YoloDepthPublisher()
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
