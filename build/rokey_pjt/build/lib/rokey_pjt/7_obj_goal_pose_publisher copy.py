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

# ========================
# 상수 정의
# ========================
YOLO_MODEL_PATH = '/home/mi/rokey_ws/model/my_best.pt'

RGB_TOPIC = 'cropped/rgb/image_raw'
DEPTH_TOPIC = 'cropped/depth/image_raw'
CAMERA_INFO_TOPIC = 'cropped/camera_info'

GOAL_POSE_TOPIC = 'yolo_goal_pose'

TARGET_CLASS_ID = 0  # YOLO 클래스 ID: 0 = car
MAX_DEPTH_METERS = 10.0
DEPTH_PATCH_SIZE = 5
# ========================

class GoalPosePublisher(Node):  # ROS2 노드 클래스
    def __init__(self):
        super().__init__('goal_pose_publisher')  # 노드 이름 지정
        self.get_logger().info("GoalPosePublisher node is starting...")
        self.bridge = CvBridge()
        self.model = YOLO(YOLO_MODEL_PATH)  # YOLO 모델 로드

        # 최신 이미지 및 정보 저장용
        self.latest_depth_image = None
        self.latest_rgb_image = None
        self.latest_rgb_msg = None
        self.K = None  # 카메라 내부 파라미터 행렬

        self.lock = threading.Lock()
        self.should_exit = False

        # TF 변환용 버퍼 및 리스너
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 센서 메시지 구독 설정
        self.create_subscription(Image, DEPTH_TOPIC, self.depth_callback, 10)
        self.create_subscription(Image, RGB_TOPIC, self.rgb_callback, 10)
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, 10)

        # 목표 포즈 발행 퍼블리셔
        self.pose_pub = self.create_publisher(PoseStamped, GOAL_POSE_TOPIC, 10)

        # YOLO 추론 및 포즈 발행 처리 루프 실행
        threading.Thread(target=self.processing_loop, daemon=True).start()

    def camera_info_callback(self, msg):
        # 카메라 내부 파라미터 (fx, fy, cx, cy) 저장
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(f"CameraInfo received: fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}")

    def depth_callback(self, msg):
        # Depth 이미지를 OpenCV 형식으로 변환
        try:
            with self.lock:
                self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {e}")

    def rgb_callback(self, msg):
        # RGB 이미지를 OpenCV 형식으로 변환
        try:
            with self.lock:
                self.latest_rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.latest_rgb_msg = msg
        except Exception as e:
            self.get_logger().error(f"RGB image conversion failed: {e}")

    def get_stable_depth(self, u, v, depth_image, patch_size=DEPTH_PATCH_SIZE, max_depth=MAX_DEPTH_METERS):
        # (u,v) 픽셀 주위에서 유효한 depth 값의 중간값(median)을 사용해 안정적인 거리 측정
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
        if z <= 0.0 or z > max_depth:
            return None, u_min, u_max, v_min, v_max
        return z, u_min, u_max, v_min, v_max

    def processing_loop(self):
        # 메인 처리 루프: RGB + Depth → YOLO → 거리 계산 → TF 변환 → goal_pose 발행
        cv2.namedWindow('YOLO + Depth + Map', cv2.WINDOW_NORMAL)
        while not self.should_exit:
            # 가장 최근 이미지 안전하게 복사
            with self.lock:
                if self.latest_rgb_image is None or self.latest_depth_image is None or self.K is None:
                    continue
                rgb_image = self.latest_rgb_image.copy()
                depth_image = self.latest_depth_image.copy()
                rgb_msg = self.latest_rgb_msg

            # Depth 이미지를 blur 처리 → 노이즈 제거
            depth_blurred = cv2.GaussianBlur(depth_image, (5, 5), 0)

            # YOLO 객체 탐지 실행
            results = self.model(rgb_image)
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    if class_id != TARGET_CLASS_ID:  # 사람 클래스만 탐지
                        continue

                    # 중심 좌표 (u,v)
                    x_center, y_center, w, h = box.xywh[0].cpu().numpy()
                    u, v = int(x_center), int(y_center)

                    # 좌표가 이미지 범위를 벗어나면 무시
                    if not (0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]):
                        continue

                    # 안정적인 depth 추출
                    z, u_min, u_max, v_min, v_max = self.get_stable_depth(u, v, depth_blurred)
                    if z is None:
                        self.get_logger().info(f"Invalid depth at ({u},{v}), skipping")
                        continue

                    # 픽셀 좌표(u,v) → 카메라 좌표(x,y,z)
                    fx, fy = self.K[0, 0], self.K[1, 1]
                    cx, cy = self.K[0, 2], self.K[1, 2]
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy

                    # PointStamped: 카메라 좌표계에서 3D 위치 지정
                    pt_camera = PointStamped()
                    pt_camera.header = rgb_msg.header
                    pt_camera.point.x = x
                    pt_camera.point.y = y
                    pt_camera.point.z = z

                    try:
                        # TF를 사용하여 카메라 좌표 → map 좌표로 변환
                        pt_map = self.tf_buffer.transform(pt_camera, 'map', timeout=rclpy.duration.Duration(seconds=0.5))
                        self.get_logger().info(f"Publish goal: map (x={pt_map.point.x:.2f}, y={pt_map.point.y:.2f}, z={pt_map.point.z:.2f})")

                        # map 기준 goal pose 메시지 생성 및 발행
                        goal_pose = PoseStamped()
                        goal_pose.header.frame_id = 'map'
                        goal_pose.header.stamp = rclpy.time.Time().to_msg()
                        goal_pose.pose.position.x = pt_map.point.x
                        goal_pose.pose.position.y = pt_map.point.y
                        goal_pose.pose.position.z = 0.0
                        goal_pose.pose.orientation.w = 1.0

                        self.pose_pub.publish(goal_pose)

                    except Exception as e:
                        self.get_logger().warn(f"TF 변환 실패: {e}")

                    # 시각화 오버레이: 박스, 중심점, 거리 등
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(rgb_image, (u, v), 5, (0, 0, 255), -1)
                    cv2.putText(rgb_image, f"{z:.2f}m", (u, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.rectangle(rgb_image, (u_min, v_min), (u_max - 1, v_max - 1), (0, 0, 255), 1)

                    break  # 첫 번째 valid 사람만 처리 후 종료

            # 이미지 표시 (OpenCV)
            cv2.imshow('YOLO + Depth + Map', rgb_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.should_exit = True
                break

    def destroy_node(self):
        # 종료 시 처리
        self.should_exit = True
        super().destroy_node()

def main():
    rclpy.init()
    node = GoalPosePublisher()
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
