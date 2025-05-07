#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
import math


def quaternion_to_yaw(qx, qy, qz, qw):
    """쿼터니언 → Yaw(라디안)"""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class CurrentPoseChecker(Node):
    def __init__(self):
        super().__init__('current_pose_checker')
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.listener_callback,
            10
        )

    def listener_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation

        yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        yaw_deg = math.degrees(yaw)

        self.get_logger().info(f'현재 위치: x={x:.2f}, y={y:.2f}, yaw={yaw_deg:.1f}도')


def main(args=None):
    rclpy.init(args=args)
    node = CurrentPoseChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
