from copy import deepcopy
import numpy as np
import pinocchio as pin
from scipy.linalg import expm
from typing import Optional, Callable
from .trajectory import (
    SegmentedCartesianTrajectory,
    CartesianSegment,
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)


class LineSegmentCartesianSpace(CartesianSegment):
    """Straight Cartesian segment between two poses."""

    def __init__(self, ee_frame_name: str, weights: TrajectoryPointWeights,
                 goal_tolerance: Optional[float] = None,
                 goal_tolerance_boost: float = 1.0,
                 goal_weight_boost: float = 1.0
                 ):
        super().__init__(ee_frame_name, weights)
        self.goal_tolerance = goal_tolerance
        self.goal_tolerance_boost = goal_tolerance_boost
        self.goal_weight_boost = goal_weight_boost
        self.w_boost = 1.0

    def interpolate_weighted_point(self, alpha, alpha_w
                                   ) -> WeightedTrajectoryPoint:
        """Interpolate one sample along the line segment."""

        translation = self.x_from + alpha * self.x_delta
        rotation = expm(self.r_delta_log * alpha) @ self.r_from
        ee_des_pos = pin.SE3(rotation, translation)

        # Approximate the desired Cartesian velocity from consecutive samples.
        dt = self.current_t - self.last_t

        if dt == 0.0:
            q = self.q0
            dq = np.zeros(self.pin_model.nv)
        else:
            ee_des_vel = (translation - self.last_x) / dt
            q, dq = self.inverse_kinematics(ee_des_pos, ee_des_vel, self.ik_q)

        self.ik_q = q
        self.last_x = translation
        self.last_t = self.current_t

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

    def evaluate_dist_to_goal(self, curr_pos, t) -> float:
        """Update finish conditions and optional weight boosting."""
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

        return dist_to_goal

    def get_traj_point_at_tq(self, t: list[np.float64], q: np.ndarray
                             ) -> list[WeightedTrajectoryPoint]:
        assert t[0] >= self.t_from

        curr_pose = self.get_end_effector_pose_from_q_as_se3(q)
        self.evaluate_dist_to_goal(curr_pose.translation, t[0])

        points = []
        last_x = None
        for one_t in t:
            self.current_t = one_t
            alpha = min((one_t - self.t_from) / self.duration
                if self.duration > 0.0 else 1.0, 1.0)
            points += [self.interpolate_weighted_point(alpha, alpha)]
            if last_x is None:
                last_x = self.last_x
        self.last_x = last_x
        self.last_t = t[0]
        return points


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
    ):
        super().__init__(x, transition_time, w_mul,
                         ee_frame_name, rotation_rpy, weights,
                         goal_tolerance, info_logger)

        self.segment = LineSegmentCartesianSpace(ee_frame_name, weights)
        self.segment.info_logger = info_logger
        self.segment.goal_tolerance_boost = goal_tolerance_boost
        self.segment.goal_weight_boost = goal_weight_boost
