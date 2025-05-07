import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped, PointStamped
import tf2_ros
import tf2_geometry_msgs
import math

class GoalPoseClient(Node):
    def __init__(self):
        super().__init__('goal_pose_client')

        self.navigator = BasicNavigator()
        self.last_goal_position = None
        self.robot_position = None
        self.task_active = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(PoseStamped, '/yolo_goal_pose', self.goal_callback, 10)

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
            base_point.header.stamp = rclpy.time.Time(seconds=0).to_msg()

            point_map = self.tf_buffer.transform(
                base_point,
                'map',
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            self.robot_position = (point_map.point.x, point_map.point.y)
        except Exception as e:
            self.get_logger().warn(f"⚠️ 로봇 위치 갱신 실패: {e}")

    def goal_callback(self, msg):
        if self.task_active:
            
            feedback = self.navigator.getFeedback()
            self.get_logger().warn(feedback)
            if feedback and feedback.distance_remaining is not None:
                    self.get_logger().info(
                        f"🟡 이동 중... 현재 목표까지 남은 거리: {feedback.distance_remaining:.2f}m")
            return

        if self.robot_position is None:
            self.get_logger().warn("⚠️ 현재 로봇 위치를 사용할 수 없습니다. 목표 무시")
            return

        rx, ry = self.robot_position
        gx, gy = msg.pose.position.x, msg.pose.position.y
        distance = math.hypot(gx - rx, gy - ry)

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

        self.last_goal_position = (gx, gy)
        self.task_active = True
        self.navigator.goToPose(msg)
        self.get_logger().info(f"🚶 이동 시작: x={gx:.2f}, y={gy:.2f}")

    def is_near_previous_goal(self, x, y, threshold=3.0):
        if self.last_goal_position is None:
            return False
        px, py = self.last_goal_position
        return math.hypot(x - px, y - py) < threshold

    def spin(self):
        while rclpy.ok():

            rclpy.spin_once(self, timeout_sec=0.1)

            if self.task_active and self.navigator.isTaskComplete():
                result = self.navigator.getResult()
                if result == TaskResult.SUCCEEDED:
                    self.get_logger().info("🎯 목표 도달 완료")
                elif result == TaskResult.CANCELED:
                    self.get_logger().warn("⚠️ 목표가 취소되었습니다.")
                elif result == TaskResult.FAILED:
                    self.get_logger().warn("⚠️ 목표 수행 실패")
                    (code, msg) = self.navigator.getTaskError()
                    self.get_logger().warn(f"에러 코드: {code} / 메세지: {msg}")
                else:
                    self.get_logger().warn(f"⚠️ 알 수 없는 결과 코드: {result}")

                self.task_active = False
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
        executor.shutdown() 
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
