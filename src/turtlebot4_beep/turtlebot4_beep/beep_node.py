import rclpy
from rclpy.node import Node
from irobot_create_msgs.msg import AudioNoteVector, AudioNote
from builtin_interfaces.msg import Duration

class BeepNode(Node):
    def __init__(self):
        super().__init__('beep_node')
        self.pub = self.create_publisher(AudioNoteVector, '/robot4/cmd_audio', 10)
        
        # 1초 후에 소리 전송
        self.timer = self.create_timer(1, self.timer_callback)

    def timer_callback(self):
        msg = AudioNoteVector()
        msg.append = False
        msg.notes = [
            AudioNote(frequency=880, max_runtime=Duration(sec=0, nanosec=300_000_000)),  # 삐
            AudioNote(frequency=440, max_runtime=Duration(sec=0, nanosec=300_000_000)),  # 뽀
            AudioNote(frequency=880, max_runtime=Duration(sec=0, nanosec=300_000_000)),  # 삐
            AudioNote(frequency=440, max_runtime=Duration(sec=0, nanosec=300_000_000)),  # 뽀
        ]

        self.get_logger().info('삐뽀삐보 소리 전송 중...')
        self.pub.publish(msg)
        self.get_logger().info('삐뽀삐보 소리 전송 완료...')
        self.timer.cancel()  # 타이머 종료

def main(args=None):
    rclpy.init(args=args)
    node = BeepNode()
    rclpy.spin_once(node, timeout_sec=3)  # 빠르게 돌리고 종료
    node.destroy_node()
    rclpy.shutdown()
