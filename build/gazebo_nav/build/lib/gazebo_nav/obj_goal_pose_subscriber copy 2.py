import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, PointStamped
import tf2_ros
import tf2_geometry_msgs
import math

class GoalPoseClient(Node):
    def __init__(self):
        super().__init__('goal_pose_client')

        self.goal_in_progress = False
        self.goal_handle = None
        self.last_goal_position = None
        self.robot_position = None  # map 기준 로봇 위치

        # TF Buffer 및 Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 액션 클라이언트
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # 토픽 구독
        self.create_subscription(PoseStamped, '/yolo_goal_pose', self.goal_callback, 10)

        # TF Tree 안정화 후 위치 갱신 타이머 설정
        self.get_logger().info("TF Tree 안정화 시작. 5초 후 위치 갱신 시작합니다.")
        self.start_timer = self.create_timer(5.0, self.start_position_update)

    def start_position_update(self):
        self.get_logger().info("TF Tree 안정화 완료. 로봇 위치 갱신 시작합니다.")
        self.position_timer = self.create_timer(0.5, self.update_robot_position)
        self.start_timer.cancel()

    def update_robot_position(self):
        try:
            base_point = PointStamped()
            base_point.header.frame_id = 'base_link'
            base_point.header.stamp = rclpy.time.Time(seconds=0).to_msg()  # 🔧 시간 0으로 설정

            point_map = self.tf_buffer.transform(
                base_point,
                'map',
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            self.robot_position = (point_map.point.x, point_map.point.y)
        except Exception as e:
            self.get_logger().warn(f"⚠️ 로봇 위치 갱신 실패: {e}")


    def goal_callback(self, msg):
        if self.goal_in_progress:
            return

        if self.robot_position is None:
            self.get_logger().warn("⚠️ 현재 로봇 위치를 사용할 수 없습니다. 목표 무시")
            return

        rx, ry = self.robot_position
        gx, gy = msg.pose.position.x, msg.pose.position.y
        distance = math.hypot(gx - rx, gy - ry)

        # 디버그 로그 출력
        now = self.get_clock().now().to_msg()
        time_str = f"{now.sec}.{str(now.nanosec).zfill(9)[:3]}"
        self.get_logger().info(
            f"[{time_str}] 목표 좌표: ({gx:.2f}, {gy:.2f}) | 로봇 좌표: ({rx:.2f}, {ry:.2f}) | 거리: {distance:.2f}m"
        )

        if distance < 2.0:
            self.get_logger().info(f"❌ 너무 가까운 목표 (거리: {distance:.2f}m). 생략")
            return
        if distance > 6.0:
            self.get_logger().info(f"❌ 너무 먼 목표 (거리: {distance:.2f}m). 생략")
            return
        if self.is_near_previous_goal(gx, gy):
            self.get_logger().info("❌ 이전 목표와 유사. 생략")
            return

        self.goal_in_progress = True
        self.last_goal_position = (gx, gy)
        self.send_goal(msg)

    def is_near_previous_goal(self, x, y, threshold=3.0):
        if self.last_goal_position is None:
            return False
        px, py = self.last_goal_position
        return math.hypot(x - px, y - py) < threshold

    def send_goal(self, pose_stamped):
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("❌ 액션 서버를 찾을 수 없습니다.")
            self.goal_in_progress = False
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose_stamped

        self.get_logger().info(
            f"🚶 이동 시작: x={pose_stamped.pose.position.x:.2f}, y={pose_stamped.pose.position.y:.2f}"
        )
        send_goal_future = self.action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        self.goal_handle = future.result()

        if not self.goal_handle.accepted:
            self.get_logger().warn("❌ 목표가 거부되었습니다.")
            self.goal_in_progress = False
            return

        self.get_logger().info("🟢 목표 수락됨. 결과 대기 중...")
        self._get_result_future = self.goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        try:
            result = future.result().result
            self.get_logger().info(f"🎯 목표 도달 완료 (상태: {future.result().status})")
        except Exception as e:
            self.get_logger().error(f"❌ 결과 처리 중 에러 발생: {e}")
        finally:
            self.goal_in_progress = False
            self.last_goal_position = None

def main():
    rclpy.init()
    node = GoalPoseClient()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
