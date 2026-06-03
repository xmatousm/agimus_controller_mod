import argparse

from agimus_msgs.msg import MpcInput, MpcInputArray

from agimus_controller_ros.ros_utils import (
    mpc_msg_to_weighted_traj_point,
)

from agimus_controller.trajectory import TrajectoryBuffer

from agimus_controller_ros.mpc_debug_mixin import MpcDebugMarkers

import rclpy
from agimus_controller_ros.agimus_controller import AgimusController
from rclpy.impl.logging_severity import LoggingSeverity
from rclpy.qos import QoSProfile, ReliabilityPolicy
from agimus_demos_common.node_utils import init_spin_node

class AgimusControllerFullHorizon(AgimusController):
    def __init__(self, node_name: str = "agimus_controller_node",
                 frame: str = "",
                 parent_frame: str = "") -> None:
        super().__init__(node_name)

        self.debug_samples = self.get_logger().is_enabled_for(LoggingSeverity.DEBUG)
        self.initialized = False
        self.logger = self.get_logger()

        self.horizon_size_full = 0
        for factor, n_step in zip(self.params.ocp.dt_factor_n_seq.factors,
                                  self.params.ocp.dt_factor_n_seq.n_steps):
            self.horizon_size_full += factor * n_step

        self.debug_markers = MpcDebugMarkers(
            self,
            frame_name=frame,
            parent_frame_name=parent_frame,
            horizon_size=self.ocp_params.horizon_size,
            horizon_size_full=self.horizon_size_full)
        self.last_removed = None


        # replace the original MPCInput subscriber with the one for MPCInputArray
        self.destroy_subscription(self.subscriber_mpc_input)
        self.subscriber_mpc_input = None

        self.subscriber_mpc_input_array = self.create_subscription(
            MpcInputArray,
            "mpc_input_array",
            self.mpc_input_array_callback,
            qos_profile=QoSProfile(
                depth=1000,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )
        self.half_run = False

    def initialization_callback(self):
        super().initialization_callback()
        if self.timer is None:
            return

        self.debug_markers.initialize(self.rmodel)

        # Change the timer to run twice per cycle, to allow skipping only half
        # a cycle (needed for proper reading of inputs)
        self.destroy_timer(self.timer)
        self.create_timer(0.5 / self.params.rate, self.run_callback)


    def buffer_has_enough_data(self, ratio: float) -> bool:
        """Return True if the buffer has enough data, False otherwise."""
        return (
                len(self.traj_buffer) * self.ocp_params.dt
                >= self.ocp_params.total_time
        )

    def mpc_input_callback(self, msg: MpcInput) -> None:
        raise NotImplementedError()

    def mpc_input_array_callback(self, msg: MpcInputArray) -> None:
        """Fill the new point msg in the trajectory buffer."""

        assert len(msg.inputs) == self.horizon_size_full + 1

        pid_from = None
        traj_buffer_in = TrajectoryBuffer(self.params.ocp.dt_factor_n_seq)
        for msg_i in msg.inputs:
            w_traj_point = mpc_msg_to_weighted_traj_point(
                msg_i, self.get_clock().now().nanoseconds
            )

            traj_buffer_in.append(w_traj_point)
            self.traj_buffer.append(w_traj_point)
            self.params.ocp.effector_frame_name = msg_i.ee_inputs[0].frame_id

            if pid_from is None:
                pid_from = w_traj_point.point.id

        pid_to = w_traj_point.point.id
        self.debug_markers.mpc_input_publish(traj_buffer_in)

        if self.debug_samples:
            self.logger.debug(f"MPC input {pid_from} - {pid_to}")

    def run_callback(self, *args) -> None:
        # TODO - do we need a lock for traj_buffer (w.r.t. input callback)?
        if self.half_run:
            self.half_run = False
            return

        len_buf = len(self.traj_buffer)
        hs = self.horizon_size_full

        assert len_buf % (hs + 1) == 0

        if len_buf < (hs + 1):
            self.logger.error("MPC: No data, skipping one half-cycle")
            return

        self.half_run = True

        # if there are more batches, use the last, remove the rest
        while len_buf > (hs + 1):
            self.logger.error(f"MPC: {self.traj_buffer[0].point.id} - " +
                              f"{self.traj_buffer[hs].point.id} " +
                              " - to much data, removing this batch")
            # remove one batch
            for i in range(hs + 1):
                self.traj_buffer.pop(0)
            len_buf = len(self.traj_buffer)

        if self.debug_samples:
            self.logger.debug(f"MPC: {self.traj_buffer[0].point.id} - " +
                             f"{self.traj_buffer[hs].point.id}")

        assert len_buf == hs + 1

        self.debug_markers.mpc_references_publish(self.traj_buffer)

        super().run_callback(*args)

        self.debug_markers.mpc_debug_data_markers_publish(
            self.mpc.mpc_debug_data)

        assert len(self.traj_buffer) == hs
        # batch is horizon_size + 1, one element from the buffer is already
        # removed by ocp, so remove the rest
        for i in range(hs):
            self.traj_buffer.pop(0)


def main(args=None) -> None:
    parser = argparse.ArgumentParser("agimus_controller_full_horizon_node")
    parser.add_argument("--frame", type=str, required=True)
    parser.add_argument("--parent-frame", type=str, required=True)

    init_spin_node(args, AgimusControllerFullHorizon, parser)

if __name__ == "__main__":
    main()
