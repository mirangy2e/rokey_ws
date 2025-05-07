import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator
import threading
import math


class GoalPoseSubscriber(Node):
    def __init__(self):
        super().__init__('goal_pose_subscriber')
        self.navigator = TurtleBot4Navigator()

        self.latest_goal_msg = None       # 가장 최근 수신된 목표 Pose
        self.last_goal_position = None    # 마지막으로 실제로 이동한 좌표
        self.goal_in_progress = False     # 현재 목표 진행 중인지 여부
        self.current_position = None      # 현재 로봇 위치
        self.navigation_lock = threading.Lock()

        # 구독: YOLO 목표 위치
        self.create_subscription(
            PoseStamped,
            '/yolo_goal_pose',
            self.goal_callback,
            10
        )
        
        # 구독: 현재 위치 (Odometry)
        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # 타이머: 1초 주기로 목표를 확인하고 실행
        self.create_timer(1.5, self.process_latest_goal)
        self.create_timer(5, self.checkTaskComplete)

    def odom_callback(self, msg):
        """로봇의 현재 위치 저장"""
        self.current_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )

    def goal_callback(self, msg):
        """목표 실행 중일 경우 토픽 무시, 그렇지 않으면 저장"""
        if self.goal_in_progress:
            return  # 이동 중이면 수신된 토픽 무시
        self.latest_goal_msg = msg

    def checkTaskComplete(self):
        if self.navigator.isTaskComplete():
            self.goal_in_progress = False
            self.get_logger().info("⚪ 네비게이션 도달 완료. 상태 초기화")

    def process_latest_goal(self):
        """주기적으로 최신 목표를 확인하고 실행"""
        if self.latest_goal_msg is None or self.goal_in_progress:
            return

        msg = self.latest_goal_msg
        original_x = msg.pose.position.x
        original_y = msg.pose.position.y

        if self.current_position is None:
            self.get_logger().warn('아직 /odom 데이터를 수신하지 못했습니다.')
            return

        robot_x, robot_y = self.current_position
        distance_to_robot = math.hypot(original_x - robot_x, original_y - robot_y)

        # 거리 조건: 너무 가까움
        if distance_to_robot < 2.0:
            self.get_logger().info(
                f'로봇과 너무 가까운 목표 (거리 {distance_to_robot:.2f}m). 생략.'
            )
            return

        # 거리 조건: 너무 멀음
        if distance_to_robot > 6.0:
            self.get_logger().info(
                f'목표 위치가 너무 멉니다. (거리: {distance_to_robot:.2f}m). 생략.'
            )
            return

        # 마지막 목표와 비슷한 위치인지 확인
        if self.is_near_previous_goal(original_x, original_y):
            return

        with self.navigation_lock:
            dx = original_x
            dy = original_y
            norm = math.hypot(dx, dy)

            if norm > 3.0:
                scale = (norm - 3.0) / norm
                new_x = dx * scale
                new_y = dy * scale
            else:
                self.get_logger().info("목표 좌표의 크기가 3m보다 작아 원본 좌표를 사용합니다.")
                new_x = original_x
                new_y = original_y

            self.get_logger().info(f"원본 좌표: x={original_x:.2f}, y={original_y:.2f}")
            self.get_logger().info(f"수정된 목표 수신: x={new_x:.2f}, y={new_y:.2f}")

            self.last_goal_position = (new_x, new_y)
            self.goal_in_progress = True
            self.latest_goal_msg = None  # 해당 목표는 처리 완료

            new_pose = PoseStamped()
            new_pose.header = msg.header
            new_pose.pose.position.x = new_x
            new_pose.pose.position.y = new_y
            new_pose.pose.position.z = msg.pose.position.z
            new_pose.pose.orientation = msg.pose.orientation

            threading.Thread(
                target=self.start_navigation, args=(new_pose,), daemon=True
            ).start()

    def is_near_previous_goal(self, x, y, threshold=3.0):
        """이전 목표와 너무 비슷하면 True"""
        if self.last_goal_position is None:
            return False
        prev_x, prev_y = self.last_goal_position
        return math.hypot(x - prev_x, y - prev_y) < threshold

    def start_navigation(self, goal_pose):
        try:
            self.get_logger().info("🟡 네비게이션 시작")
            self.navigator.startToPose(goal_pose)

        except RuntimeError as e:
            self.get_logger().error(f"Navigation runtime error: {e}")
        except Exception as e:
            self.get_logger().error(f"Navigation error: {e}")
        finally:
            self.goal_in_progress = False
            self.get_logger().info("⚪ 네비게이션 도달 완료. 상태 초기화")




def main():
    rclpy.init()
    node = GoalPoseSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
