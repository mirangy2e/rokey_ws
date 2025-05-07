#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator
from tf_transformations import quaternion_from_euler
from builtin_interfaces.msg import Time  # 누락됨
import time



def create_pose(x, y, yaw_deg, navigator):
    """x, y, yaw(도 단위) → PoseStamped 생성"""
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = navigator.get_clock().now().to_msg()
    pose.pose.position.x = x
    pose.pose.position.y = y

    yaw_rad = yaw_deg * 3.141592 / 180.0
    q = quaternion_from_euler(0, 0, yaw_rad)
    pose.pose.orientation.x = q[0]
    pose.pose.orientation.y = q[1]
    pose.pose.orientation.z = q[2]
    pose.pose.orientation.w = q[3]
    return pose


def main():
    rclpy.init()

    # 두 navigator 인스턴스 생성
    dock_navigator = TurtleBot4Navigator()
    nav_navigator = BasicNavigator(node_name='navigator_robot4')

    # 1. 초기 위치 설정
    initial_pose = create_pose(-0.01, -0.01, 0.0, nav_navigator)  # NORTH
    nav_navigator.setInitialPose(initial_pose)
    nav_navigator.get_logger().info(f'초기 위치 설정 중...')
    time.sleep(1.0) #AMCL이 초기 pose 처리 시 필요한 시간과 TF를 얻을 수 있게 됨
    nav_navigator.waitUntilNav2Active()

    # 2. 도킹되어 있다면 언도킹
    if dock_navigator.getDockedStatus():
        dock_navigator.get_logger().info('현재 도킹 상태 → 언도킹 시도')
        dock_navigator.undock()
    else:
        dock_navigator.get_logger().info('언도킹 상태에서 시작함')

    # 3. 목표 위치 이동 명령
    goal_pose = create_pose(-0.87, -1.21, 90.0, nav_navigator)  # EAST
    nav_navigator.goToPose(goal_pose)

    # 4. 이동 중 피드백 표시
    while not nav_navigator.isTaskComplete():
        feedback = nav_navigator.getFeedback()
        if feedback:
            remaining = feedback.distance_remaining
            nav_navigator.get_logger().info(f'남은 거리: {remaining:.2f} m')

    # 5. 결과 확인
    result = nav_navigator.getResult()
    if result == TaskResult.SUCCEEDED:
        nav_navigator.get_logger().info('목표 위치 도달 성공')
        dock_navigator.dock()
        dock_navigator.get_logger().info('도킹 요청 완료')
    elif result == TaskResult.CANCELED:
        nav_navigator.get_logger().warn('이동이 취소되었습니다.')
    elif result == TaskResult.FAILED:
        error_code, error_msg = nav_navigator.getTaskError()
        nav_navigator.get_logger().error(f'이동 실패: {error_code} - {error_msg}')
    else:
        nav_navigator.get_logger().warn('알 수 없는 결과 코드 수신')

    # 6. 종료
    dock_navigator.destroy_node()
    nav_navigator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
