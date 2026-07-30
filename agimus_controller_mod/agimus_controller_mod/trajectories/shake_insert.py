from copy import deepcopy
import numpy as np
import pinocchio as pin

from .trajectory import (
    CartesianSegment,
    TrajectoryPoint,
    WeightedTrajectoryPoint,
)


class ShakeInsert(CartesianSegment):
    """Going down with shaking."""

    def __init__(self, ee_frame_name: str):
        super().__init__(ee_frame_name)

        self.shake_amount = None
        self.delta_z = None
        self.duration = None

    def init_segment(self):
        super().init_segment()

        assert self.duration > 0.0
        assert self.shake_amount is not None
        assert self.delta_z is not None

        self.t_to = self.t_from + self.duration
        self.x_to = self.x_from
        self.r_to = self.r_from

    def interpolate_weighted_point(self, alpha, alpha_w
                                   ) -> WeightedTrajectoryPoint:
        """Interpolate one sample along the saw-tooth segment."""

        # keep rotation as is
        rotation = self.r_from

        # circular motion in the x-y plane
        a = alpha * 2 * np.pi
        dx = np.cos(a) * self.shake_amount
        dy = np.sin(a) * self.shake_amount

        translation = self.x_from + np.array([dx, dy, self.delta_z])
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

        traj_weights = deepcopy(self.weights)
        return WeightedTrajectoryPoint(
            point=deepcopy(traj_point), weights=traj_weights)

    def evaluate_stopping_criterion(self, t: float, q: np.ndarray) -> None:
        """Finishing criterion - time only."""
        if t >= self.t_to:
            self.running = False
