from abc import ABC
import numpy as np
import pinocchio as pin
from scipy.linalg import logm
from typing import Optional

from .trajectory import Trajectory, TrajectoryPointWeights


class TrajectorySegment(Trajectory, ABC):
    """Base class for a single segment of a piece-wise trajectory."""

    def __init__(self, ee_frame_name: str):
        super().__init__(ee_frame_name)
        self.t_from = None
        self.t_to = None
        self.x_len_init = None
        self.x_from = None
        self.x_to = None
        self.x_delta = None
        self.r_from = None
        self.r_to = None
        self.duration = None
        self.velocity = None
        self.ee_init_pos = None
        self.running = False
        self.goal_min_dist = None
        self.last_x = None
        self.last_t = 0.0
        self.current_t = 0.0
        self.r_delta_log = None

        self.weights = TrajectoryPointWeights()

        # the pose weight is interpolated between these two, if given
        self.w_pose_from: Optional[np.ndarray] = None
        self.w_pose_to: Optional[np.ndarray] = None

    def initialize(self, pin_model: pin.Model, q0: np.ndarray) -> None:
        """Initialize the trajectory generator."""
        super().initialize(pin_model, q0)
        self.ee_init_pos = self.get_end_effector_pose_from_q_as_se3(self.q0)
        self.last_x = self.ee_init_pos.translation.copy()
        self.last_t = 0.0

    def initialize_w(self, weights: TrajectoryPointWeights) -> None:
        self.w_pose_from = None
        self.w_pose_to = None
        self.weights = weights

    def init_segment(self) -> None:
        """Additional initialization of the segment in a derived class.
        Called from set_segment()."""
        pass

    def set_segment(self, t: np.float64,
                    x_from: np.ndarray,
                    x_to: np.ndarray,
                    r_from: np.ndarray,
                    r_to: np.ndarray,
                    duration: Optional[float] = None,
                    velocity: Optional[float] = None,
                    weights: Optional[TrajectoryPointWeights] = None,
                    w_pose_from: Optional[np.ndarray] = None,
                    w_pose_to: Optional[np.ndarray] = None,
                    ):
        """Initialize the segment end-points, timing, and weightds."""

        self.running = True
        self.x_from = x_from
        self.x_to = x_to
        self.x_delta = x_to - x_from
        self.x_len_init = np.linalg.norm(self.x_delta)
        self.r_from = r_from
        self.r_to = r_to
        self.r_delta_log = logm(r_to @ r_from.T)

        self.velocity = velocity

        # Either duration or velocity must be given and positive (or both).
        # The missing one of these is computed using the other one and the
        # cartesian distance. If both are given, the duration is updated so
        # that it can lead to a slower speed, but not to a faster.

        if velocity is None:
            # velocity from duration (can be computed as zero here)
            assert duration is not None and duration > 0.0
            self.duration = duration
            self.velocity = self.x_len_init / duration

        else:
            assert velocity > 0.0
            # compute duration from the velocity, use the maximum of it and
            # the given one; the result must be positive
            # (so for a zero-length trajectory, the positive duration must be
            # given)
            self.duration = np.linalg.norm(self.x_delta) / velocity
            if duration is not None:
                self.duration = max(self.duration, duration)
            assert self.duration > 0.0

        self.t_from = t
        self.t_to = t + self.duration

        self.w_pose_from = w_pose_from
        self.w_pose_to = w_pose_to

        if weights is not None:
            self.weights = weights

        assert (self.w_pose_from is None) == (self.w_pose_to is None)
        self.init_segment()
