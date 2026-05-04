import argparse
from typing import Optional
import rclpy

def init_spin_node(args, node_class: type,
                   arg_parser: Optional[argparse.ArgumentParser] = None):
    node = None
    try:
        rclpy.init(args=args)

        if arg_parser is not None:
            args = rclpy.utilities.remove_ros_args(args)
            args = arg_parser.parse_args(args[1:])  # skip the script name
            node = node_class(**vars(args))
        else:
            node = node_class()

        if hasattr(node, "do_work"):
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.0)
                node.do_work()

        else:
            rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok:
            rclpy.shutdown()
