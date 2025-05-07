import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped, PointStamped
import tf2_ros
import tf2_geometry_msgs
import math

# 상수 정의
MIN_GOAL_DISTANCE = 0.5  # 최소 유효 거리 (m)
MAX_GOAL_DISTANCE = 6.0  # 최대 유효 거리 (m)

class GoalPoseSubscriber(Node):
    def __init__(self):
        super().__init__('goal_pose_subscriber')

        # Nav2 기본 네비게이터 객체 생성
        self.navigator = BasicNavigator()
        self.last_goal_position = None  # 이전 목표 좌표 저장
        self.robot_position = None      # 현재 로봇 위치 (map 기준)
        self.task_active = False        # 현재 이동 중 여부

        # TF 변환 설정 (base_link → map)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # YOLO 탐지 결과로부터 목표 위치 수신
        self.create_subscription(PoseStamped, '/yolo_goal_pose', self.goal_callback, 10)

        # TF Tree 초기화 대기 후 위치 갱신 타이머 시작
        self.get_logger().info("노드 시작. 목표 좌표 수신 대기 중...")
        self.get_logger().info("TF Tree 안정화 대기 중...")
        self.start_timer = self.create_timer(5.0, self.start_position_update)

        # 이동 상태 확인 주기적 타이머
        self.feedback_timer = self.create_timer(0.5, self.check_task_status)

    def start_position_update(self):
        # TF 안정화 후 로봇 위치 주기적 갱신 타이머 시작
        self.get_logger().info("TF Tree 안정화 완료. 로봇 위치 갱신 시작")
        self.position_timer = self.create_timer(0.5, self.update_robot_position)
        self.start_timer.cancel()

    def update_robot_position(self):
        # base_link → map 좌표 변환하여 현재 로봇 위치 갱신
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
            self.get_logger().warn(f"로봇 위치 갱신 실패: {e}")

    def goal_callback(self, msg):
        # 목표가 이미 진행 중이면 무시
        if self.task_active:
            return

        # 현재 위치와 목표 위치 거리 계산
        rx, ry = self.robot_position
        gx, gy = msg.pose.position.x, msg.pose.position.y
        distance = math.hypot(gx - rx, gy - ry)

        self.get_logger().info(
            f"목표 좌표: ({gx:.2f}, {gy:.2f}) | 로봇 좌표: ({rx:.2f}, {ry:.2f}) | 거리: {distance:.2f}m"
        )

        # 너무 가까운 또는 먼 목표는 무시
        if distance < MIN_GOAL_DISTANCE:
            self.get_logger().info(f"너무 가까운 목표 (거리: {distance:.2f}m). 생략")
            return
        if distance > MAX_GOAL_DISTANCE:
            self.get_logger().info(f"너무 먼 목표 (거리: {distance:.2f}m). 생략")
            return
        # if self.is_near_previous_goal(gx, gy):
        #     self.get_logger().info("이전 목표와 유사. 생략")
        #     return

        # 유효한 목표로 이동 시작
        self.last_goal_position = (gx, gy)
        self.task_active = True
        try:
            self.navigator.goToPose(msg)
            self.get_logger().info(f"이동 시작: x={gx:.2f}, y={gy:.2f}")
        except Exception as e:
            self.get_logger().warn(f"이동 중 예외 발생: {e}. 새 목표 대기 중...")
            self.task_active = False
            self.last_goal_position = None

    def is_near_previous_goal(self, x, y, threshold=3.0):
        # 이전 목표와의 거리 비교
        if self.last_goal_position is None:
            return False
        px, py = self.last_goal_position
        return math.hypot(x - px, y - py) < threshold

    def check_task_status(self):
        # 이동 중일 때 Nav2 상태 확인
        if self.task_active:
            try:
                if self.navigator.isTaskComplete():
                    result = self.navigator.getResult()
                    if result == TaskResult.SUCCEEDED:
                        self.get_logger().info("목표 도달 완료")
                    elif result == TaskResult.CANCELED:
                        self.get_logger().warn("목표가 취소되었습니다.")
                    elif result == TaskResult.FAILED:
                        self.get_logger().warn("목표 수행 실패")
                    else:
                        self.get_logger().warn(f"알 수 없는 결과 코드: {result}")

                    self.task_active = False
                    self.last_goal_position = None
                else:
                    feedback = self.navigator.getFeedback()
                    if feedback and feedback.distance_remaining is not None:
                        self.get_logger().info(
                            f"이동 중... 현재 목표까지 남은 거리: {feedback.distance_remaining:.2f}m")
            except Exception as e:
                self.get_logger().warn(f"상태 확인 중 오류 발생. 예외: {e}. 목표 대기 중...")
                self.task_active = False
                self.last_goal_position = None


def main():
    rclpy.init()
    node = GoalPoseSubscriber()
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
