from typing import Optional
import numpy as np
import pinocchio as pin
from rclpy.impl.rcutils_logger import RcutilsLogger

from .trajectory import (
    Trajectory,
    get_all_weights,
    WeightedTrajectoryPoint,
    TrajectoryPointWeights,
    trajectory_parameters
)

from .line_segment_cartesian_space import LineSegmentCartesianSpace


class LineCartesianSpace(Trajectory):
    """Trajectory of poly-line defined by end-points in cartesian space."""

    def __init__(
            self,
            x,
            transition_time,
            w_mul,
            ee_frame_name: str,
            rotation_rpy,
            weights: TrajectoryPointWeights,
            goal_tolerance=None,
            goal_tolerance_boost=1.0,
            goal_weight_boost=1.0,
            logger: Optional[RcutilsLogger] = None,
    ):

        super().__init__(ee_frame_name)

        assert len(rotation_rpy) == 3, "rotation_rpy length must be 3"
        assert len(x) > 0 and len(x) % 3 == 0, \
            "x length must be a multiple of 3"

        self.x = np.array(x).reshape((-1, 3))
        self.n_points = len(self.x)

        assert len(transition_time) == self.n_points + 1, \
            "time length must be the number of points + 1"

        self.transition_time = transition_time

        if w_mul is None or len(w_mul) <= 1:
            self.w_mul = [1.0] * self.n_points
        else:
            assert len(w_mul) == self.n_points, \
                "w_mul length must be the number of points"
            self.w_mul = w_mul

        if goal_tolerance is None or len(goal_tolerance) <= 1:
            self.goal_tolerance = [None] * self.n_points
        else:
            assert len(goal_tolerance) == self.n_points, \
                "goal_tolerance length must be the number of points"
            self.goal_tolerance = goal_tolerance

        self.weights = weights
        self.w_pose = weights.w_end_effector_poses[ee_frame_name]
        self.ee_init_pos = None
        self.point = -1  # the current point we are moving to

        self.segment = LineSegmentCartesianSpace(ee_frame_name)
        self.segment.logger = logger
        self.rotation = pin.rpy.rpyToMatrix(
            rotation_rpy[0], rotation_rpy[1], rotation_rpy[2])
        self.goal_tolerance = goal_tolerance
        if self.goal_tolerance is None:
            self.goal_tolerance = [None] * self.n_points
        self.goal_tolerance_boost = goal_tolerance_boost
        self.goal_weight_boost = goal_weight_boost
        self.logger = logger

    @classmethod
    def from_params(cls,
                    params: trajectory_parameters.Params,
                    nq: int,
                    ee_frame_name: str,
                    logger: RcutilsLogger):

        weights = get_all_weights(params, nq, ee_frame_name)

        return LineCartesianSpace(
            x=params.line_endpoints.x,
            transition_time=params.line_endpoints.time,
            w_mul=params.line_endpoints.w_mul,
            ee_frame_name=ee_frame_name,
            rotation_rpy=params.line_endpoints.rotation,
            weights=weights,
            goal_tolerance=params.line_endpoints.goal_tolerance,
            goal_tolerance_boost=params.line_endpoints.goal_tolerance_boost,
            goal_weight_boost=params.line_endpoints.goal_weight_boost,
            logger=logger,
        )

    def initialize(self, pin_model: pin.Model, q0: np.ndarray) -> None:
        """Initialize the trajectory generator."""
        super().initialize(pin_model, q0)
        self.ee_init_pos = self.get_end_effector_pose_from_q_as_se3(self.q0)

        self.segment.initialize(pin_model, q0)
        self.segment.initialize_w(self.weights)
        self.segment.goal_weight_boost = self.goal_weight_boost
        self.segment.goal_tolerance_boost = self.goal_tolerance_boost
        self.point = -1

    def get_traj_point_at_tq(self, t: list[np.float64], q: np.ndarray
                             ) -> list[WeightedTrajectoryPoint]:
        if not self.segment.running:  # switch the segment
            if self.point < 0:
                self.segment.goal_tolerance = self.goal_tolerance[0]
                self.segment.set_segment(
                    t=t[0],
                    x_from=self.ee_init_pos.translation,
                    x_to=self.x[0],
                    r_from=self.ee_init_pos.rotation,
                    r_to=self.rotation,
                    duration=self.transition_time[0],
                    w_pose_from=self.w_pose * self.w_mul[0],
                    w_pose_to=self.w_pose * self.w_mul[0])
                self.point = 0
            else:
                point_from = self.point
                self.point = (self.point + 1) % self.n_points

                self.segment.goal_tolerance = self.goal_tolerance[self.point]
                self.segment.set_segment(
                    t=t[0],
                    x_from=self.x[point_from],
                    x_to=self.x[self.point],
                    r_from=self.rotation,
                    r_to=self.rotation,
                    duration=self.transition_time[point_from + 1],
                    w_pose_from=self.w_pose * self.w_mul[point_from],
                    w_pose_to=self.w_pose * self.w_mul[self.point])

            if self.logger is not None:
                self.logger.info(f"Point set: {self.point}, " +
                                 f"t={self.segment.duration}, " +
                                 f"x_to={self.segment.x_to}")

        # interpolate cartesian line
        return self.segment.get_traj_point_at_tq(t, q)
