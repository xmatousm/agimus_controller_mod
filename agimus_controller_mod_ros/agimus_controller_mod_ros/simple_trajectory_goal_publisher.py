import rclpy
from agimus_controller_mod_msgs.action import TrajectoryAction
from rclpy.node import Node
from rclpy.action import ActionClient
from .node_utils import init_spin_node

from .trajectory_builders.trajectory_builder import (
    get_trajectory_builder,
    SegmentedCartesianTrajectory,
    trajectory_parameters,
)


class SimpleTrajectoryGoalPublisher(Node):
    """Example client that sends trajectory segment goals to a server."""

    def __init__(self):
        super().__init__("simple_trajectory_goal_publisher")
        self._action_client = ActionClient(self, TrajectoryAction,
                                           'trajectory_goal')

        self.param_listener = trajectory_parameters.ParamListener(self)
        self.params = self.param_listener.get_params()
        self._id: int = -1
        self.croco_nq = 7

        self.builder = get_trajectory_builder(self.params.trajectory_name,
                                              self.get_logger())

        trajectory = self.builder.from_params(self.params,
                                              self.croco_nq,
                                              self.params.ee_frame_name,
                                              self.get_logger())

        assert isinstance(trajectory, SegmentedCartesianTrajectory)

        self.trajectory: SegmentedCartesianTrajectory = trajectory
        self.rotation_rpy = self.params.line_endpoints.rotation

    def do_work(self):
        """Send the next waypoint as an action goal and wait for completion."""

        # prepare action goal from the next segment
        self._id += 1
        action_goal = TrajectoryAction.Goal()
        goal = action_goal.goal
        goal.id = self._id

        # The trajectory is not initilized with robot model, etc., we are using
        # it for switching segments and generating goals only.

        self.trajectory.segment.running = False
        self.trajectory.switch_segment(0)

        self.builder.to_goal(self.trajectory.segment, goal)
        goal.rot_rpy = self.rotation_rpy  # TODO put to the builder

        # wait for server to be reade
        self.get_logger().info("Wait for server")
        self._action_client.wait_for_server()

        # send the goal
        self.get_logger().info(
            f"Sending goal {self._id}, point: {self.trajectory.point}")

        goal_future = self._action_client.send_goal_async(action_goal)
        rclpy.spin_until_future_complete(self, goal_future)
        goal_handle = goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        # wait for the result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        self.get_logger().info(f'Result: {result}')


def main(args=None):
    init_spin_node(args, SimpleTrajectoryGoalPublisher)


if __name__ == "__main__":
    main()
