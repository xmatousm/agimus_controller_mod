from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import logm
import pinocchio as pin
from typing import Optional, Callable

from agimus_controller.trajectories.trajectory_base import TrajectoryBase
from agimus_controller.trajectory import (
    TrajectoryPoint,
    WeightedTrajectoryPoint,
    TrajectoryPointWeights
)


class Trajectory(TrajectoryBase, ABC):
    """Common base for trajectory generators."""

    def __init__(self, ee_frame_name) -> None:
        super().__init__(ee_frame_name)
        self.info_logger: Optional[Callable] = None

    @abstractmethod
    def get_traj_point_at_tq(
            self, t: list[np.float64], q: np.ndarray
    ) -> list[WeightedTrajectoryPoint]:
        """List of weighted trajectory points for a list of times and the
        current robot configuration.
        """

    def get_traj_point_at_t(self, t: np.float64) -> WeightedTrajectoryPoint:
        raise NotImplementedError()

    def inverse_kinematics(
            self,
            ee_des_pos: pin.SE3,
            ee_des_vel: np.ndarray,
            ref_q: np.ndarray,
            precision=1e-5,
            it_max=10000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inverse kinematics to reach the desired end effector pose."""
        i = 0
        success = False
        ik_q = ref_q.copy()
        while True:
            ik_ee_pose = self.get_end_effector_pose_from_q_as_se3(ik_q)
            dMi = ee_des_pos.actInv(ik_ee_pose)
            error = pin.log(dMi).vector[:3]
            if np.linalg.norm(error) < precision:
                success = True
                break
            if i > it_max:
                break

            pin.computeJointJacobians(self.pin_model, self.pin_data, ik_q)
            jaco_ee = pin.getFrameJacobian(
                self.pin_model,
                self.pin_data,
                self.ee_frame_id,
                pin.ReferenceFrame.LOCAL,
            )[:3, :]
            dq = -jaco_ee.T @ np.linalg.solve(jaco_ee @ jaco_ee.T, error)
            ik_q[:] = pin.integrate(self.pin_model, ik_q, dq)
            i += 1

        if not success:
            error_msgs = (
                f"Inverse kinematics 6D failed to converge with error: "
                f"{error}. Number of iteration: {i}"
            )
            raise RuntimeError(error_msgs)

        pin.forwardKinematics(self.pin_model, self.pin_data, ik_q)
        pin.updateFramePlacement(self.pin_model, self.pin_data,
                                 self.ee_frame_id)
        pin.computeJointJacobians(self.pin_model, self.pin_data, ik_q)
        jaco_ee = pin.getFrameJacobian(
            self.pin_model,
            self.pin_data,
            self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )[:3, :]
        dq = jaco_ee.T @ np.linalg.solve(jaco_ee @ jaco_ee.T, ee_des_vel[:3])

        return ik_q, dq.copy()


class CartesianSegment(Trajectory, ABC):
    """Base class for a segment of a piecewise trajectory in Cartesian space."""

    def __init__(self, ee_frame_name: str, weights: TrajectoryPointWeights):
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
        self.last_x = None
        self.last_t = 0.0
        self.current_t = 0.0
        self.r_delta_log = None

        self.weights = weights
        self.ik_q = None

        # the pose weight is interpolated between these two, if given
        self.w_pose_from: Optional[np.ndarray] = None
        self.w_pose_to: Optional[np.ndarray] = None

    def initialize(self, pin_model: pin.Model, q0: np.ndarray) -> None:
        """Initialize the trajectory generator."""

        super().initialize(pin_model, q0)
        self.ik_q = q0.copy()

        self.ee_init_pos = self.get_end_effector_pose_from_q_as_se3(self.q0)
        self.last_x = self.ee_init_pos.translation.copy()
        self.last_t = 0.0

    def init_segment(self) -> None:
        """Additional derived class setup called at the end of set_segment()."""
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
        """Configure segment end points, timing, and optional pose weights."""

        self.running = True
        self.x_from = x_from
        self.x_to = x_to
        self.x_delta = x_to - x_from
        self.x_len_init = np.linalg.norm(self.x_delta)
        self.r_from = r_from
        self.r_to = r_to
        self.r_delta_log = logm(r_to @ r_from.T)

        self.velocity = velocity

        # Either duration or velocity must be provided.
        # If one is missing, derive it from the other and the Cartesian length.
        # When both are provided, the longer duration wins, so the commanded
        # speed never exceeds the requested velocity.

        if velocity is None:
            # velocity from duration (can be computed as zero here)
            assert duration is not None and duration > 0.0
            self.duration = duration
            self.velocity = self.x_len_init / duration

        else:
            assert velocity > 0.0
            # Compute duration from the velocity, use the maximum of it and
            # the given one; the result must be positive (so for a zero-length
            # trajectory, the positive duration must be given).
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


class SegmentedCartesianTrajectory(Trajectory, ABC):
    """Base class for piece-wise trajectories composed of Cartesian segments."""

    def __init__(
            self,
            x,
            transition_time,
            w_mul,
            ee_frame_name: str,
            rotation_rpy,
            weights: TrajectoryPointWeights,
            goal_tolerance,
            info_logger: Optional[Callable] = None,
    ) -> None:

        super().__init__(ee_frame_name)

        self.segment = None
        self.x = None
        self.n_points = 0
        self.transition_time = None
        self.rotation = None
        self.w_mul = None
        self.weights = weights
        self.w_pose = weights.w_end_effector_poses[ee_frame_name]
        self.goal_tolerance = None
        self.point = -1  # the current point we are moving to
        self.info_logger = info_logger

        assert len(rotation_rpy) == 3, "rotation length must be 3"
        self.rotation = pin.rpy.rpyToMatrix(
            rotation_rpy[0], rotation_rpy[1], rotation_rpy[2])

        assert len(x) > 0 and len(x) % 3 == 0, "x length must be multiple of 3"

        self.x = np.array(x).reshape((-1, 3))
        self.n_points = len(self.x)

        # init pos for the case the trajectory is not initialized, so that
        # switch segment would still work
        self.ee_init_pos = pin.SE3(self.rotation, self.x[0])

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

    def initialize(self, pin_model: pin.Model, q0: np.ndarray) -> None:
        """Initialize the trajectory generator."""

        super().initialize(pin_model, q0)
        self.segment.initialize(pin_model, q0)

        self.ee_init_pos = self.get_end_effector_pose_from_q_as_se3(self.q0)
        self.point = -1

    def switch_segment(self, t):
        """Activate the next segment in the sequence."""

        if self.point < 0:
            # The first segment starts from the initial end-effector pose.
            self.segment.goal_tolerance = self.goal_tolerance[0]
            self.segment.set_segment(
                t=t,
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
            # Later segments connect consecutive configured waypoints and loop.
            self.point = (self.point + 1) % self.n_points

            self.segment.goal_tolerance = self.goal_tolerance[self.point]
            self.segment.set_segment(
                t=t,
                x_from=self.x[point_from],
                x_to=self.x[self.point],
                r_from=self.rotation,
                r_to=self.rotation,
                duration=self.transition_time[point_from + 1],
                w_pose_from=self.w_pose * self.w_mul[point_from],
                w_pose_to=self.w_pose * self.w_mul[self.point])

        if self.info_logger is not None:
            self.info_logger(f"Point set: {self.point}, " +
                             f"t={self.segment.duration}, " +
                             f"x_to={self.segment.x_to}")

    def get_traj_point_at_tq(self, t: list[np.float64], q: np.ndarray
                             ) -> list[WeightedTrajectoryPoint]:
        # Advance to the next waypoint once the active segment completes.
        if not self.segment.running:
            self.switch_segment(t[0])

        # Delegate interpolation to the currently active segment instance.
        return self.segment.get_traj_point_at_tq(t, q)
