import numpy as np
from agimus_demos_common.trajectories.line_segment_cartesian_space import \
    LineSegmentCartesianSpace
from scipy.linalg import logm
from agimus_controller.trajectory import WeightedTrajectoryPoint


class LineSegmentCartesianSpaceAdaptive(LineSegmentCartesianSpace):
    """Line segment defined by end-points in the cartesian space."""

    def __init__(self, ee_frame_name: str, dt: float):
        super().__init__(ee_frame_name)
        self.dt = dt

    def init_segment(self) -> None:
        super().init_segment()

        # an adaptive segment must have the goal tolerance and nonzero velocity
        assert self.goal_tolerance is not None

        if self.velocity == 0.0:
            self.velocity = 0.1  # TODO parameter somewhere?

    def get_traj_point_at_tq(self, t: list[np.float64], q: np.ndarray
                             ) -> list[WeightedTrajectoryPoint]:
        assert t[0] >= self.t_from

        curr_pose = self.get_end_effector_pose_from_q_as_se3(q)
        dist_to_goal = self.evaluate_dist_to_goal(curr_pose.translation, t[0])

        # update the 'from' pose w.r.t the actual configuration
        self.x_from  = curr_pose.translation
        self.x_delta = self.x_to - self.x_from
        duration = np.linalg.norm(self.x_delta) / self.velocity

        # TODO solve when we are ath the goal (duration=0)

        self.r_from = curr_pose.rotation
        self.r_delta_log = logm(self.r_to @ self.r_from.T)

        points = []
        t0 = t[0] - 10 * self.dt
        # TODO check this (current is now, the next point should be for
        #  now + dt)
        for one_t in t:
            self.current_t = one_t
            alpha = min((one_t - t0) / duration, 1.0)
            if self.x_len_init == 0.0:
                alpha_w = min((one_t - self.t_from) / self.duration, 1.0)
            else:
                alpha_w = 1.0 - min(dist_to_goal / self.x_len_init, 1.0)
            points += [self.interpolate_weighted_point(alpha, alpha_w)]

        self.last_t = t[0]
        return points
