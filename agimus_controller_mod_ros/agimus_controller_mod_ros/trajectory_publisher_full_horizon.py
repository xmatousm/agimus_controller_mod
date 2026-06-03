import time
import numpy as np
from agimus_controller_ros.ros_utils import (
    weighted_traj_point_to_mpc_msg,
)
from rclpy.impl.logging_severity import LoggingSeverity

from rclpy.qos import QoSProfile, ReliabilityPolicy

from agimus_msgs.msg import MpcDebug, MpcInputArray

from agimus_controller.trajectory import TrajectoryPointWeights

from agimus_demos_common.trajectory_weights_parameters import (
    trajectory_weights_params,
)

from agimus_demos_common.trajectories.line_cartesian_space import \
    LineCartesianSpace

from agimus_demos_common.trajectories.line_cartesian_space_adaptive import \
    LineCartesianSpaceAdaptive

from agimus_controller_ros.simple_trajectory_publisher import (
    TrajectoryPublisherBase,
)

from agimus_demos_common.simple_trajectory_publisher_mod import (
    OcpParamsClientMixin
)

from agimus_demos_common.trajectories.trajectory_base_mod import (
    get_weights,
    TrajectoryBaseMod
)

from agimus_demos_common.node_utils import init_spin_node


class TrajectoryPublisherFullHorizon(TrajectoryPublisherBase,
                                     OcpParamsClientMixin):

    def __init__(self):
        self.initialized = False
        TrajectoryPublisherBase.__init__(self,
                                         "trajectory_publisher_full_horizon")
        OcpParamsClientMixin.__init__(self, self)

        self.debug = self.get_logger().is_enabled_for(LoggingSeverity.DEBUG)

        self.param_listener = trajectory_weights_params.ParamListener(self)

        self.params = self.param_listener.get_params()
        self.ee_frame_name = self.params.ee_frame_name
        self._id: int = 0
        self.t: np.float64 = np.float64(0.0)
        self.max_delay = self.params.max_delay

        self.last_mpc_point_id = None

        self.croco_nq = 7
        self.use_q = False  # send current q to trajectory

        self.trajectory = self.get_trajectory(self.params)

        self._mpc_debug_sub = self.create_subscription(
            MpcDebug,
            "mpc_debug",
            self.mpc_debug_callback,
            qos_profile=QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )

        # replace the original MPCInput publisher with the one for MPCInputArray
        self.destroy_publisher(self.publisher_)

        self.publisher_ = self.create_publisher(
            MpcInputArray,
            "mpc_input_array",
            qos_profile=QoSProfile(
                depth=1000,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )

        self.first_run = True
        self.initialized = True
        self.get_logger().info("Initialized.")

    def ready_callback(self):
        if not self.initialized:
            # Base can run this via the timer before our init finishes
            self.get_logger().warn("Not ready.")
            self.destroy_timer(self.timer)
            self.timer = self.create_timer(0.1, self.ready_callback)
            return

        self.destroy_timer(self.timer)

        self.get_logger().info("Ready.")
        self.timer = self.create_timer(self.dt, self.publish_mpc_input)

    def mpc_debug_callback(self, msg: MpcDebug):
        if self.debug:
            self.get_logger().debug(f"MPC last: {msg.trajectory_point_id}")
        self.last_mpc_point_id = msg.trajectory_point_id

    def get_trajectory(self,
                       params: trajectory_weights_params.Params) -> TrajectoryBaseMod:
        """Build the chosen trajectory."""

        if params.trajectory_name in ("line_cartesian_space",
                                      "line_cartesian_space_adaptive",
                                      ):

            weights = TrajectoryPointWeights(
                w_robot_configuration=get_weights(
                    params.w_q, self.croco_nq),
                w_robot_velocity=get_weights(
                    params.w_qdot, self.croco_nq),
                w_robot_acceleration=get_weights(
                    params.w_qddot, self.croco_nq),
                w_robot_effort=get_weights(
                    params.w_robot_effort, self.croco_nq),
                w_end_effector_poses={
                    self.ee_frame_name: get_weights(self.params.w_pose, 6)
                }
            )

            if params.trajectory_name == "line_cartesian_space":
                return LineCartesianSpace(
                    x=params.line_endpoints.x,
                    transition_time=params.line_endpoints.time,
                    w_mul=params.line_endpoints.w_mul,
                    ee_frame_name=self.ee_frame_name,
                    rotation_rpy=params.line_endpoints.rotation,
                    weights=weights,
                    goal_tolerance=params.line_endpoints.goal_tolerance,
                    goal_tolerance_boost=params.line_endpoints.goal_tolerance_boost,
                    goal_weight_boost=params.line_endpoints.goal_weight_boost,
                    logger=self.get_logger(),
                )

            elif params.trajectory_name == "line_cartesian_space_adaptive":
                return LineCartesianSpaceAdaptive(
                    dt=self.dt,
                    x=params.line_endpoints.x,
                    transition_time=params.line_endpoints.time,
                    w_mul=params.line_endpoints.w_mul,
                    ee_frame_name=self.ee_frame_name,
                    rotation_rpy=params.line_endpoints.rotation,
                    weights=weights,
                    goal_tolerance=params.line_endpoints.goal_tolerance,
                    goal_tolerance_boost=params.line_endpoints.goal_tolerance_boost,
                    goal_weight_boost=params.line_endpoints.goal_weight_boost,
                    logger=self.get_logger(),
                )



        else:
            raise ValueError("Unknown Trajectory " + params.trajectory_name)

    def publish_mpc_input(self):
        if self.first_run:
            self.get_logger().info("Running.")
            self.first_run = False
            self.trajectory.initialize(self.robot_models.robot_model, self.q0)

        if self.debug:
            t0 = time.time_ns() / 1e9
            self.get_logger().debug(
                f"Run: {self._id}  / {self.last_mpc_point_id}")

        #if self.last_mpc_point_id is not None and self.max_delay > 0:
        #    delay = self._id - self.last_mpc_point_id - 1
        #    if delay > self.max_delay:
        #        self.get_logger().error(
        #            f"{self._id}: Input to MPC delay: {delay}; skipping one cycle.")
        #        return

        point_id = self._id
        t_list = [self.t + self.dt * i for i in range(self.horizon_size_full1)]
        points = self.trajectory.get_traj_point_at_tq(t_list, self.current_q)

        mpc_input_array = MpcInputArray()

        num = len(points)
        i = 0
        for w_traj_point in points:
            i += 1
            alpha = i / num
            w_traj_point.weights.w_end_effector_poses[self.ee_frame_name] *= alpha

            w_traj_point.point.id = point_id
            point_id += 1
            msg = weighted_traj_point_to_mpc_msg(w_traj_point)
            mpc_input_array.inputs += [msg]

        self.publisher_.publish(mpc_input_array)

        self._id += 1
        self.t += self.dt

        if self.debug:
            self.get_logger().debug(f"  Done. {time.time_ns() / 1e9 - t0}")


def main(args=None):
    init_spin_node(args, TrajectoryPublisherFullHorizon)


if __name__ == "__main__":
    main()
