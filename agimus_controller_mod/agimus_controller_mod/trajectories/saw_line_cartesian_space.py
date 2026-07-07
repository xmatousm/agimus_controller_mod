from typing import Optional, Callable
from copy import deepcopy
import numpy as np
import pinocchio as pin
from scipy.linalg import expm

from .trajectory import (
    SegmentedCartesianTrajectory,
    CartesianLineSegment,
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)


class SawLineSegmentCartesianSpace(CartesianLineSegment):
    """Cartesian segment with a repeated saw-tooth offset."""

    def __init__(self, ee_frame_name: str):
        super().__init__(ee_frame_name)

        self.tooth_length = None
        self.tooth_length_rel = None
        self.tooth_tip = None

    def init_segment(self) -> None:
        super().init_segment()

        assert self.tooth_length > 0.0
        assert len(self.tooth_tip) == 3

        # Precompute the tooth length in normalized segment coordinates.
        d = np.linalg.norm(self.x_to - self.x_from)
        self.tooth_length_rel = self.tooth_length / d

    def interpolate_weighted_point(self, alpha, alpha_w
                                   ) -> WeightedTrajectoryPoint:
        """Interpolate one sample along the saw-tooth segment."""

        translation_line = self.x_from + alpha * self.x_delta
        rotation = expm(self.r_delta_log * alpha) @ self.r_from

        # Resolve the current tooth index and normalized phase within it.
        saw_n = alpha // self.tooth_length_rel
        saw_t = (alpha % self.tooth_length_rel) / self.tooth_length_rel

        alpha_tooth = saw_n * self.tooth_length_rel
        t0 = self.x_from + alpha_tooth * self.x_delta
        t1 = t0 + self.tooth_tip
        t2 = self.x_from + (alpha_tooth + self.tooth_length_rel) * self.x_delta

        if saw_t < 0.5:
            translation = t0 + (t1 - t0) * saw_t * 2
        else:
            translation = t1 + (t2 - t1) * (saw_t - 0.5) * 2

        ee_des_pos = pin.SE3(rotation, translation)

        # Approximate the desired Cartesian velocity from consecutive samples.
        dt = self.current_t - self.last_t

        if dt == 0.0:
            q = self.q0
            dq = np.zeros(self.pin_model.nv)
        else:
            ee_des_vel = (translation - self.last_x) / dt
            q, dq = self.inverse_kinematics(ee_des_pos, ee_des_vel, self.last_q)

        self.last_q = q
        self.last_x = translation

        ddq = np.zeros(self.pin_model.nv)
        u = pin.rnea(self.pin_model, self.pin_data, q, dq, ddq)

        traj_point = TrajectoryPoint(
            robot_configuration=q,
            robot_velocity=dq,
            robot_acceleration=ddq,
            robot_effort=u,
            end_effector_poses={
                self.ee_frame_name: pin.SE3ToXYZQUAT(ee_des_pos)},
        )

        # Optionally ramp the pose weight over the segment.
        traj_weights = deepcopy(self.weights)
        if self.w_pose_from is not None:
            w_pose = self.w_pose_from * (1 - alpha_w) + self.w_pose_to * alpha_w
            traj_weights.w_end_effector_poses[self.ee_frame_name] = w_pose

        return WeightedTrajectoryPoint(
            point=deepcopy(traj_point), weights=traj_weights)

    def evaluate_stopping_criterion(self, t: float, q: np.ndarray):
        """Finishing criterion - time only."""
        if t >= self.t_to:
            self.running = False

class SawLineCartesianSpace(SegmentedCartesianTrajectory):
    """Piecewise Cartesian trajectory whose segments follow a saw pattern."""

    def __init__(
            self,
            x,
            transition_time,
            w_mul,
            ee_frame_name: str,
            rotation_rpy,
            weights: TrajectoryPointWeights,
            tooth_length: float,
            tooth_tip: float,
            info_logger: Optional[Callable] = None,
    ):
        super().__init__(x, transition_time, w_mul,
                         ee_frame_name, rotation_rpy, weights,
                         info_logger)

        self.segment = SawLineSegmentCartesianSpace(ee_frame_name)
        self.segment.weights = weights
        self.segment.info_logger = info_logger
        self.segment.tooth_length = tooth_length
        self.segment.tooth_tip = tooth_tip
