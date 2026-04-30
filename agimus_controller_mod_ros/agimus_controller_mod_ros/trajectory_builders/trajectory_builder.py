from abc import ABC, abstractmethod
import numpy as np
import importlib
from typing import Union
from agimus_controller_mod_ros.trajectory_parameters import \
    trajectory_parameters
from agimus_controller.trajectory import TrajectoryPointWeights

from agimus_controller_mod.trajectories.trajectory import (
    Trajectory,
    CartesianSegment,
    SegmentedCartesianTrajectory,
)

from rclpy.impl.rcutils_logger import RcutilsLogger

from agimus_controller_mod_msgs.msg import TrajectoryGoal


class TrajectoryBuilder(ABC):
    """Base class for a builder of a trajectory or a trajectory segment."""

    @abstractmethod
    def from_params(self,
                    params: trajectory_parameters.Params,
                    nq: int,
                    ee_frame_name: str,
                    logger: RcutilsLogger) -> Trajectory:
        """Build a trajectory from trajectory parameters."""

    @abstractmethod
    def from_goal(self,
                  goal: TrajectoryGoal,
                  nq: int) -> CartesianSegment:
        """Build a trajectory segment from a trajectory goal."""

    @abstractmethod
    def to_goal(self, trajectory: Trajectory,
                goal: TrajectoryGoal) -> TrajectoryGoal:
        """Build a trajectory goal from a trajectory segment."""


def get_weights(weights: list[np.float64], size: int) -> np.ndarray:
    """Normalize a weight parameter to a NumPy array of length ``size``.

    A scalar-like list such as ``[1.0]`` is broadcast to all entries.
    """
    if len(weights) == 1:
        return np.array(weights * size)
    else:
        assert len(weights) == size
        return np.array(weights)


def get_all_weights(params: Union[trajectory_parameters.Params, TrajectoryGoal],
                    nq: int,
                    ee_frame_name: str,
                    ) -> TrajectoryPointWeights:
    """Build a ``TrajectoryPointWeights`` object from ROS params or a goal."""
    # TrajectoryGoal and Params have the same weight attributes
    return TrajectoryPointWeights(
        w_robot_configuration=get_weights(params.w_q, nq),
        w_robot_velocity=get_weights(params.w_qdot, nq),
        w_robot_acceleration=get_weights(params.w_qddot, nq),
        w_robot_effort=get_weights(params.w_robot_effort, nq),
        w_end_effector_poses={
            ee_frame_name: get_weights(params.w_pose, 6)
        }
    )

def set_all_weights(weights: TrajectoryPointWeights,
                    goal: TrajectoryGoal,
                    ee_frame_name: str,
                    ) -> None:
    """Set weights in the ROS goal from TrajectoryPointWeights object."""

    goal.w_q = list(weights.w_robot_configuration)
    goal.w_qdot = list(weights.w_robot_velocity)
    goal.w_qddot = list(weights.w_robot_acceleration)
    goal.w_robot_effort = list(weights.w_robot_effort)
    goal.w_pose = list(weights.w_end_effector_poses[ee_frame_name])


def get_trajectory_builder(trajectory_name: str,
                           logger: RcutilsLogger,
                           ) -> TrajectoryBuilder:
    """Instantiate the builder referenced as a 'module:Class' string."""

    trajectory_name_list = trajectory_name.split(':')
    if len(trajectory_name_list) != 2:
        logger.error(f'Wrong trajectory "{trajectory_name}".')
        raise ValueError(f'Wrong trajectory "{trajectory_name}".')

    # Split the user-facing identifier into the relative module and class name.
    module_name, class_name = trajectory_name.split(':')
    package_name = __name__.rpartition(".")[0]
    module_name = f'{package_name}.{module_name}'
    logger.info(f'Importing {module_name}')

    # Import the module dynamically (so new builders can be added).
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        logger.error(f'Wrong trajectory "{trajectory_name}".')
        raise

    logger.info(f'Getting {class_name} from {module_name}')

    try:
        BuilderClass = getattr(module, class_name)
    except AttributeError:
        logger.error(f'Wrong trajectory "{trajectory_name}".')
        raise

    if not issubclass(BuilderClass, TrajectoryBuilder):
        logger.error(f'Wrong trajectory "{trajectory_name}".')
        raise ValueError(f'The class {module_name}.{class_name} is not ' +
                         'subclass of TrajectoryBuilder')

    return BuilderClass()
