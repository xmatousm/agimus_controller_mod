from copy import deepcopy
import numpy as np
import pinocchio as pin
from scipy.linalg import expm
from typing import Optional, Callable
from .trajectory import (
    SegmentedCartesianTrajectory,
    CartesianLineSegment,
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)


class LineSegmentCartesianSpace(CartesianLineSegment):
    """Straight Cartesian segment between two poses."""

    def __init__(self, ee_frame_name: str):
        super().__init__(ee_frame_name)
        self.goal_tolerance = None
        self.goal_tolerance_boost = None
        self.goal_weight_boost = None
        self.w_boost = 1.0
        self.reg_q = None

    def interpolate_weighted_point(self, alpha, alpha_w
                                   ) -> WeightedTrajectoryPoint:
        """Interpolate one sample along the line segment."""

        translation = self.x_from + alpha * self.x_delta
        rotation = expm(self.r_delta_log * alpha) @ self.r_from
        ee_des_pos = pin.SE3(rotation, translation)

        # Approximate the desired Cartesian velocity from consecutive samples.
        dt = self.current_t - self.last_t

        if dt == 0.0:
            q = self.last_q
            dq = np.zeros(self.pin_model.nv)
        else:
            ee_des_vel = (translation - self.last_x) / dt
            q, dq = self.inverse_kinematics(ee_des_pos, ee_des_vel, self.last_q,
                                            reg_q=self.reg_q)

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

        # Goal-based boosting is applied on top of the nominal/interpolated
        # pose weights.
        traj_weights.w_end_effector_poses[self.ee_frame_name] *= self.w_boost

        return WeightedTrajectoryPoint(
            point=deepcopy(traj_point), weights=traj_weights)

    def evaluate_stopping_criterion(self, t: float, q: np.ndarray):
        """Update finish conditions and optional weight boosting."""
        pose = self.get_end_effector_pose_from_q_as_se3(q)
        curr_pos = pose.translation
        dist_to_goal = np.sqrt(np.sum((self.x_to - curr_pos) ** 2))

        # optionally boost weights when approaching the goal
        if self.goal_tolerance is not None:
            # Inside the boosted tolerance band, scale weights linearly from
            # 1.0 up to ``goal_weight_boost`` as we approach the target.
            a = max(0.0, 1.0 - dist_to_goal / (self.goal_tolerance *
                                               self.goal_tolerance_boost))

            self.w_boost = a * (self.goal_weight_boost - 1.0) + 1.0
            if self.info_logger is not None and a > 0.0:
                self.info_logger(
                    f"  Goal boost: {self.w_boost}  {dist_to_goal} {self.goal_tolerance} {self.goal_tolerance_boost}",
                    throttle_duration_sec=1.0)

        # finishing criterion - time
        if t >= self.t_to:
            if self.goal_tolerance is not None:
                # additional criterion - goal tolerance
                if dist_to_goal < self.goal_tolerance:
                    self.running = False
                else:
                    if self.info_logger is not None:
                        self.info_logger(
                            f"Dist to goal: {dist_to_goal} > {self.goal_tolerance}",
                            throttle_duration_sec=1.0)

            else:
                # no goal tolerance, the segment is finished based on time only
                self.running = False


class LineCartesianSpace(SegmentedCartesianTrajectory):
    """Piecewise-linear Cartesian trajectory through configured waypoints."""

    def __init__(
            self,
            x,
            transition_time,
            w_mul,
            ee_frame_name: str,
            rotation_rpy,
            weights: TrajectoryPointWeights,
            goal_tolerance: Optional[list] = None,
            goal_tolerance_boost: float = 1.0,
            goal_weight_boost: float = 1.0,
            info_logger: Optional[Callable] = None,
            reg_q: Optional[list] = None,
    ):
        super().__init__(x, transition_time, w_mul,
                         ee_frame_name, rotation_rpy, weights,
                         info_logger, reg_q=reg_q)

        self.segment = LineSegmentCartesianSpace(ee_frame_name)

        if goal_tolerance is None or len(goal_tolerance) <= 1:
            self.goal_tolerance = [None] * self.n_points
        else:
            assert len(goal_tolerance) == self.n_points, \
                "goal_tolerance length must be the number of points"
            self.goal_tolerance = goal_tolerance

        self.segment.weights = weights
        self.segment.reg_q = reg_q
        self.segment.info_logger = info_logger
        self.segment.goal_tolerance_boost = goal_tolerance_boost
        self.segment.goal_weight_boost = goal_weight_boost

    def switch_segment(self, t):
        super().switch_segment(t)
        self.segment.goal_tolerance = self.goal_tolerance[self.point]

