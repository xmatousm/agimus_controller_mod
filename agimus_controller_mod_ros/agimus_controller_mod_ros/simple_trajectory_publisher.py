import numpy as np
from rclpy.task import Future
from agimus_controller_ros.ros_utils import (
    weighted_traj_point_to_mpc_msg,
    get_param_from_node,
)

from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.node import Node

from agimus_msgs.msg import MpcDebug

from agimus_controller_mod_ros.trajectory_parameters import \
    trajectory_parameters

from agimus_controller_ros.simple_trajectory_publisher import (
    TrajectoryPublisherBase,
)

from .trajectories.trajectory import Trajectory

from agimus_demos_common.node_utils import init_spin_node
import importlib


class OcpParamsClientMixin:
    def __init__(self, node: Node):
        # main clock period
        self.dt = get_param_from_node(
            node, "agimus_controller_node", "ocp.dt"
        ).double_value

        # horizon size
        n_steps = get_param_from_node(
            node, "agimus_controller_node", "ocp.dt_factor_n_seq.n_steps"
        ).integer_array_value

        factors = get_param_from_node(
            node, "agimus_controller_node", "ocp.dt_factor_n_seq.factors"
        ).integer_array_value

        self.horizon_size_full1 = 1  # OCP needs +1 sample
        for factor, n_step in zip(factors, n_steps):
            self.horizon_size_full1 += factor * n_step

        node.get_logger().info(
            f"Params from OCP: dt = {self.dt}," +
            f" horizon size full (+1) = {self.horizon_size_full1}")


class SimpleTrajectoryPublisherMod(TrajectoryPublisherBase,
                                   OcpParamsClientMixin):
    """This is a modified simple trajectory publisher."""

    def __init__(self):
        self.initialized = False
        TrajectoryPublisherBase.__init__(self, "simple_trajectory_publisher")
        OcpParamsClientMixin.__init__(self, self)

        self.param_listener = trajectory_parameters.ParamListener(self)

        self.params = self.param_listener.get_params()
        self.ee_frame_name = self.params.ee_frame_name
        self._id: int = 0
        self.t: np.float64 = np.float64(0.0)
        self.max_delay = self.horizon_size_full1 + self.params.max_delay

        self.last_mpc_point_id = None

        self.croco_nq = 7
        self.future_init_done = Future()
        self.future_trajectory_done = Future()
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
        self.last_mpc_point_id = msg.trajectory_point_id

    def get_trajectory(self,
                       params: trajectory_parameters.Params) -> Trajectory:
        """Build the chosen trajectory."""

        # dynamic import of a trajectory class
        module_name, class_name = params.trajectory_name.split(':')
        package_name = self.__module__.rpartition(".")[0]
        module_name = f'{package_name}.trajectories.{module_name}'
        self.get_logger().info(f'Importing {module_name}')

        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            self.get_logger().error('Wrong trajectory ' +
                                    params.trajectory_name)
            raise

        self.get_logger().info(f'Creating {class_name} from {module_name}')

        try:
            BuilderClass = getattr(module, class_name)
        except AttributeError:
            self.get_logger().error('Wrong trajectory ' +
                                    params.trajectory_name)
            raise

        return BuilderClass.from_params(params, self.croco_nq,
                                        self.ee_frame_name,
                                        self.get_logger())

    def publish_mpc_input(self):
        if self.first_run:
            self.get_logger().info("Running.")
            self.first_run = False
            self.trajectory.initialize(self.robot_models.robot_model, self.q0)
            self.future_init_done.set_result(True)

        if self.last_mpc_point_id is not None:
            delay = self._id - self.last_mpc_point_id - 1
            if delay > self.max_delay:
                self.get_logger().error(
                    f"{self._id}: Input to MPC delay: {delay}; skipping one cycle.")
                return

        w_traj_point = self.trajectory.get_traj_point_at_tq([self.t],
                                                            self.current_q)[0]
        w_traj_point.point.id = self._id
        msg = weighted_traj_point_to_mpc_msg(w_traj_point)
        self._id += 1

        self.publisher_.publish(msg)
        if self.trajectory.trajectory_is_done:
            self.future_trajectory_done.set_result(True)
        self.t += self.dt


def main(args=None):
    init_spin_node(args, SimpleTrajectoryPublisherMod)


if __name__ == "__main__":
    main()
