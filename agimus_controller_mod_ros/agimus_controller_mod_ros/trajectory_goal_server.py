import numpy as np
import rclpy
from rclpy.action import ActionServer
from rclpy.qos import QoSProfile, ReliabilityPolicy
import pinocchio as pin

from agimus_controller_mod_msgs.action import TrajectoryAction
from agimus_msgs.msg import MpcDebug

from agimus_demos_common.node_utils import init_spin_node
from .trajectory_builders.trajectory_builder import get_trajectory_builder
import time

from agimus_controller_ros.simple_trajectory_publisher import (
    TrajectoryPublisherBase,
)

from .simple_trajectory_publisher import OcpParamsClientMixin

from rclpy.task import Future

from agimus_controller_ros.ros_utils import (
    weighted_traj_point_to_mpc_msg,
)


class TrajectoryGoalServer(TrajectoryPublisherBase,
                           OcpParamsClientMixin):
    """Action server that converts trajectory goals into MPC reference points."""

    def __init__(self):
        self.future_base_init_done = Future()
        TrajectoryPublisherBase.__init__(self, 'trajectory_goal_server')
        OcpParamsClientMixin.__init__(self, self)

        # Allow some lag between the outgoing stream and the MPC consumer
        # before intentionally skipping a publish cycle.
        self.point_delta = int(self.horizon_size_full1 * 1.3)  # TODO
        self.get_logger().info(f"Used point delta: {self.point_delta}")

        self.last_mpc_point_id = None

        self._mpc_debug_sub = self.create_subscription(
            MpcDebug,
            "mpc_debug",
            self.mpc_debug_callback,
            qos_profile=QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )

        self.get_logger().info(f'Waiting for base init')

        rclpy.spin_until_future_complete(self, self.future_base_init_done)
        self.get_logger().info(f'Starting action server')

        self.goal_done = Future()
        self.last_w_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.last_x_from = None
        self.last_r_from = None
        self.tr0 = self.get_clock().now().nanoseconds / 1e9
        self.tt0 = time.time_ns() / 1e9
        self.t: np.float64 = 0.0
        self.trajectory = None
        self.w_traj_point = None
        self._point_id = 0
        self.timer = self.create_timer(self.dt, self.publish_mpc_input)
        self.current_pose = None
        self.builder = {}

        self._action_server = ActionServer(
            self,
            TrajectoryAction,
            'trajectory_goal',
            self.execute_callback)

    def mpc_debug_callback(self, msg: MpcDebug):
        """Track the latest point id consumed by MPC."""
        self.last_mpc_point_id = msg.trajectory_point_id

    def ready_callback(self):
        """Unblock startup once the base publisher has finished initializing."""
        self.get_logger().error(f'ready')
        self.future_base_init_done.set_result(True)

    def publish_mpc_input(self):
        """Publish the next weighted trajectory point for the active goal."""
        if self.trajectory is None and self.w_traj_point is None:
            return

        delay = None
        if self.last_mpc_point_id is not None:
            delay = self._point_id - self.last_mpc_point_id
            if delay > self.point_delta:
                self.get_logger().error(
                    f"{self._point_id}: Input to MPC delay: {delay}; skipping one cycle.")
                return

        if self.trajectory is not None:
            self.w_traj_point = self.trajectory.get_traj_point_at_tq(
                [self.t], self.current_q)[0]
            self.t += self.dt
            if not self.trajectory.running:
                # Cache the final pose so execute_callback() can report the
                # terminal Cartesian error after the goal completes.
                self.goal_done.set_result(True)
                self.current_pose = self.trajectory.get_end_effector_pose_from_q_as_se3(
                    self.current_q)
                self.trajectory = None

        self.w_traj_point.point.id = self._point_id
        self._point_id += 1
        msg = weighted_traj_point_to_mpc_msg(self.w_traj_point)
        self.publisher_.publish(msg)

    def execute_callback(self, goal_handle):
        """Build a Cartesian segment from the goal and stream it to MPC."""
        action_goal: TrajectoryAction.Goal = goal_handle.request

        goal = action_goal.goal

        self.get_logger().info(
            f'Executing goal {goal.trajectory_type} {goal.id} ({goal.duration}s)')

        # The action goal names the builder to use for the segment type.
        if goal.trajectory_type in self.builder:
            builder = self.builder[goal.trajectory_type]
        else:
            try:
                builder = get_trajectory_builder(goal.trajectory_type, self.get_logger())
                self.builder[goal.trajectory_type] = builder
            except Exception:
                self.get_logger().error(f'Cannot create a goal.')
                goal_handle.abort()
                self._action_server.destroy()
                self.destroy_node()
                raise

        nq = 7  # TODO
        segment = builder.from_goal(goal, nq)
        segment.initialize(self.robot_models.robot_model, self.q0)

        rotation = pin.rpy.rpyToMatrix(
            goal.rot_rpy[0], goal.rot_rpy[1], goal.rot_rpy[2])

        if self.last_x_from is None:
            # The first goal starts from the measured robot pose. Later goals
            # continue from the previously commanded endpoint.
            self.last_x_from = segment.ee_init_pos.translation
            self.last_r_from = segment.ee_init_pos.rotation

        segment.set_segment(
            t=self.t,
            x_from=self.last_x_from,
            x_to=np.array(goal.pose),
            r_from=self.last_r_from,
            r_to=rotation,
            duration=goal.duration if goal.duration > 0.0 else None,
            velocity=goal.speed if goal.speed > 0.0 else None,
            w_pose_from=self.last_w_pose, w_pose_to=np.array(goal.w_pose),
        )

        segment.info_logger = self.get_logger().info

        self.last_w_pose = np.array(goal.w_pose)
        self.last_x_from = np.array(goal.pose)
        self.last_r_from = np.array(rotation)

        self.goal_done = Future()
        self.trajectory = segment
        rclpy.spin_until_future_complete(self, self.goal_done)
        err = np.sqrt(np.sum((self.current_pose.translation - goal.pose) ** 2))

        self.get_logger().info(f'Goal finished {goal.id} ({err})')

        result = TrajectoryAction.Result()

        goal_handle.succeed()

        result.distance = err
        result.id = goal.id
        return result


def main(args=None):
    init_spin_node(args, TrajectoryGoalServer)


if __name__ == '__main__':
    main()
