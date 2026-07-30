from abc import ABC, abstractmethod
from datetime import datetime
import pickle
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
    """Common base for trajectory generators in cartesian space."""

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
            damp=1e-2,
            conv: float = 1.0,
            reg_q: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Iterative solution of under-determined inverse kinematics for the
        end-effector pose.

        Damped Gauss-Newton method is used each step is computed as a minimum
        norm update. In each step, if regularizator is given, the norm between
        the updated configuration and regularizator is minimized. Otherwise,
        the update norm is minimized (i.e., the regularizator is effectively the
        configuration from the previous step).

        The robot must be redundant (or fully determined), and the seed must
        not be a singular configuration, i.e., the jacobian must have full row
        rank.

        :param ee_des_pos: Desired end-effector pose.
        :param ee_des_vel: Desired end-effector velocity;
                           3-vecter with position only.
        :param ref_q: Initial robot configuration used as the seed.
        :param precision: Convergence threshold for the norm of the log-pose
                          error.
        :param it_max: Maximum number of iterations.
        :param damp: Factor for damped pseudoinverse.
        :param conv: Step-size multiplier, applied to each update.
        :param reg_q: Regularizator, optional.

        :returns: The converged joint configuration and corresponding joint
        velocity.

        :raises: RuntimeError: If the iterative loop does not converge.
        """
        i = 0
        success = False
        ik_q = ref_q.copy()
        error = np.inf

        damp_mat = damp * np.eye(6)
        while i <= it_max:
            i += 1

            ik_ee_pose = self.get_end_effector_pose_from_q_as_se3(ik_q)
            dMi = ee_des_pos.actInv(ik_ee_pose)
            error = pin.log(dMi).vector

            if np.linalg.norm(error) < precision:
                success = True
                break

            pin.computeJointJacobians(self.pin_model, self.pin_data, ik_q)
            jaco_ee = pin.getFrameJacobian(
                self.pin_model,
                self.pin_data,
                self.ee_frame_id,
                pin.ReferenceFrame.LOCAL,
            )

            # damped minimum-norm solution of J dq + error = 0
            jjt = jaco_ee @ jaco_ee.T
            dq = -jaco_ee.T @ np.linalg.solve(jjt + damp_mat, error)

            # regularization - moves in the null space of the Jacobian
            # towards req_q
            if reg_q is not None:
                rq: np.ndarray = reg_q - ik_q

                # no damping here, otherwise would not converge
                dq += rq - jaco_ee.T @ np.linalg.solve(jjt, jaco_ee @ rq)

            ik_q = pin.integrate(self.pin_model, ik_q, conv * dq)

        if not success:
            error_msgs = (
                f"Inverse kinematics 6D failed to converge, iterations: {i}\n"
                f"error: {error},\n"
                f"ref_q: {np.round(ref_q * 180 / np.pi, 2)},\n",
                f"cur_q: {np.round(ik_q * 180 / np.pi, 2)},\n",
                f"desired pose: {ee_des_pos}."
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"ikt_fail_{timestamp}.pckl", "wb") as fail_file:
                pickle.dump({
                    "ref_q": ref_q,
                    "ik_q": ik_q,
                    "ee_des_pos": ee_des_pos,
                    "ee_des_vel": ee_des_vel,
                    "pin_model": self.pin_model,
                    "pin_data": self.pin_data,
                    "ee_frame_id": self.ee_frame_id,
                }, fail_file)

            raise RuntimeError(error_msgs)

        pin.forwardKinematics(self.pin_model, self.pin_data, ik_q)
        pin.updateFramePlacement(self.pin_model, self.pin_data,
                                 self.ee_frame_id)
        pin.computeJointJacobians(self.pin_model, self.pin_data, ik_q)

        # des_vel is only pose (3-vector)
        jaco_ee = pin.getFrameJacobian(
            self.pin_model,
            self.pin_data,
            self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )[:3, :]
        dq = jaco_ee.T @ np.linalg.solve(jaco_ee @ jaco_ee.T, ee_des_vel[:3])

        return ik_q, dq.copy()


class CartesianSegment(Trajectory, ABC):
    def __init__(self, ee_frame_name) -> None:
        super().__init__(ee_frame_name)

        self.ee_init_pos = None
        self.x_from = None
        self.r_from = None
        self.t_from = None
        self.x_to = None
        self.r_to = None
        self.t_to = None
        self.duration = None
        self.velocity = None
        self.reg_q = None

        self.weights = None
        self.running = False
        self.last_t = 0.0
        self.current_t = 0.0
        self.last_x = None
        self.last_q = None

        self.w_pose_from = None
        self.w_pose_to = None

    def initialize(self, pin_model: pin.Model, q0: np.ndarray) -> None:
        """Initialize the trajectory generator."""

        super().initialize(pin_model, q0)
        self.last_q = q0.copy()

        self.ee_init_pos = self.get_end_effector_pose_from_q_as_se3(self.q0)
        self.last_x = self.ee_init_pos.translation.copy()
        self.last_t = 0.0
        self.current_t = 0.0

    def init_segment(self) -> None:
        """Segment initialization after its data has been set."""

        self.running = True

        assert self.weights is not None
        assert self.x_from is not None
        assert self.r_from is not None
        assert self.t_from is not None
        self.last_t = self.t_from

    @abstractmethod
    def evaluate_stopping_criterion(self, t: float, q: np.ndarray) -> None:
        raise NotImplementedError()

    @abstractmethod
    def interpolate_weighted_point(self, alpha, alpha_w
                                   ) -> WeightedTrajectoryPoint:
        raise NotImplementedError()

    def get_traj_point_at_tq(self, t: list[np.float64], q: np.ndarray
                             ) -> list[WeightedTrajectoryPoint]:
        assert t[0] >= self.t_from

        self.evaluate_stopping_criterion(t[0], q)

        points = []
        last_x = None
        self.last_q = q
        for one_t in t:
            self.current_t = one_t
            alpha = min((one_t - self.t_from) / self.duration
                        if self.duration > 0.0 else 1.0, 1.0)
            points += [self.interpolate_weighted_point(alpha, alpha)]
            self.last_t = self.current_t
            if last_x is None:
                last_x = self.last_x
        self.last_x = last_x
        self.last_t = t[0]
        return points


class CartesianLineSegment(CartesianSegment, ABC):
    """Base class for a segment of a piecewise trajectory in Cartesian space."""

    def __init__(self, ee_frame_name: str):
        super().__init__(ee_frame_name)
        self.x_len_init = None
        self.x_delta = None
        self.r_delta_log = None

        # the pose weight is interpolated between these two, if given
        self.w_pose_from: Optional[np.ndarray] = None
        self.w_pose_to: Optional[np.ndarray] = None

    def init_segment(self) -> None:
        """Segment initialization after its data has been set."""

        super().init_segment()
        assert self.x_to is not None
        assert self.r_to is not None

        self.x_delta = self.x_to - self.x_from
        self.x_len_init = np.linalg.norm(self.x_delta)
        self.r_delta_log = logm(self.r_to @ self.r_from.T)

        # Either duration or velocity must be provided.
        # If one is missing, derive it from the other and the Cartesian length.
        # When both are provided, the longer duration wins, so the commanded
        # speed never exceeds the requested velocity.

        if self.velocity is None:
            # velocity from duration (can be computed as zero here)
            assert self.duration is not None and self.duration > 0.0
            self.velocity = self.x_len_init / self.duration

        else:
            assert self.velocity > 0.0
            # Compute duration from the velocity, use the maximum of it and
            # the given one (for a zero-length segment, the result can be zero)
            duration = np.linalg.norm(self.x_delta) / self.velocity
            if self.duration is not None:
                self.duration = max(self.duration, duration)
            else:
                self.duration = duration

            assert self.duration >= 0.0

        self.t_to = self.t_from + self.duration

        assert (self.w_pose_from is None) == (self.w_pose_to is None)


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
            info_logger: Optional[Callable] = None,
            reg_q: Optional[list] = None,
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
        self.reg_q = None
        if reg_q is not None and len(reg_q) > 1:
            assert len(reg_q) == weights.w_robot_configuration.shape[0]
            self.reg_q = np.array(reg_q)

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

    def initialize(self, pin_model: pin.Model, q0: np.ndarray) -> None:
        """Initialize the trajectory generator."""

        super().initialize(pin_model, q0)
        self.segment.initialize(pin_model, q0)

        self.ee_init_pos = self.get_end_effector_pose_from_q_as_se3(self.q0)
        self.point = -1

    def switch_segment(self, t):
        """Activate the next segment in the sequence.
        The init_segment is not called here to allow overriding. It must be then
        called later."""

        self.segment.reg_q = self.reg_q
        self.segment.velocity = None
        if self.point < 0:
            # The first segment starts from the initial end-effector pose.
            self.segment.t_from = t
            self.segment.x_from = self.ee_init_pos.translation
            self.segment.x_to = self.x[0]
            self.segment.r_from = self.ee_init_pos.rotation
            self.segment.r_to = self.rotation
            self.segment.duration = self.transition_time[0]
            self.segment.w_pose_from = self.w_pose * self.w_mul[0]
            self.segment.w_pose_to = self.w_pose * self.w_mul[0]
            self.point = 0
        else:
            point_from = self.point
            # Later segments connect consecutive configured waypoints and loop.
            self.point = (self.point + 1) % self.n_points

            self.segment.t_from = t
            self.segment.x_from = self.x[point_from]
            self.segment.x_to = self.x[self.point]
            self.segment.r_from = self.rotation
            self.segment.r_to = self.rotation
            self.segment.duration = self.transition_time[point_from + 1]
            self.segment.w_pose_from = self.w_pose * self.w_mul[point_from]
            self.segment.w_pose_to = self.w_pose * self.w_mul[self.point]

        if self.info_logger is not None:
            self.info_logger(f"Point set: {self.point}, " +
                             f"t={self.segment.duration}, " +
                             f"x_to={self.segment.x_to}")

    def get_traj_point_at_tq(self, t: list[np.float64], q: np.ndarray
                             ) -> list[WeightedTrajectoryPoint]:
        # Advance to the next waypoint once the active segment completes.
        if not self.segment.running:
            self.switch_segment(t[0])
            self.segment.init_segment()

        # Delegate interpolation to the currently active segment instance.
        return self.segment.get_traj_point_at_tq(t, q)
