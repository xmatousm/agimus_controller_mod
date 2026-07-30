from rclpy.impl.rcutils_logger import RcutilsLogger

from .trajectory_builder import (
    get_all_weights,
    set_goal,
    trajectory_parameters,
    TrajectoryGoal,
    TrajectoryBuilder,
)

import agimus_controller_mod.trajectories.shake_insert as traj


class ShakeInsert(TrajectoryBuilder):
    """Builder for ShakeInsert trajectory."""

    def from_params(self,
                    params: trajectory_parameters.Params,
                    nq: int,
                    ee_frame_name: str,
                    logger: RcutilsLogger):
        """Not implemented, building of a "full-trajectory" makes no sense."""
        raise NotImplementedError

    def from_goal(self,
                  goal: TrajectoryGoal,
                  nq: int) -> traj.ShakeInsert:
        """Build a ShakeInsert trajectory from an action goal."""
        segment = traj.ShakeInsert(goal.frame_name)

        segment.duration = goal.duration
        segment.weights = get_all_weights(goal, nq, goal.frame_name)
        segment.delta_z = goal.s1
        segment.shake_amount = goal.s2
        return segment

    def to_goal(self, segment: traj.ShakeInsert,
                goal: TrajectoryGoal) -> TrajectoryGoal:
        set_goal(goal, segment)
        goal.trajectory_type = __name__.rpartition(".")[-1] + ':' + \
                               self.__class__.__name__

        goal.s1 = segment.delta_z
        goal.s2 = segment.shake_amount

        return goal
