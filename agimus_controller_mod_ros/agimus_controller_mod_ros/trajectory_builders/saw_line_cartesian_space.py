from rclpy.impl.rcutils_logger import RcutilsLogger
import numpy as np

from .trajectory_builder import (
    get_all_weights,
    set_segment,
    set_goal,
    trajectory_parameters,
    TrajectoryGoal,
    TrajectoryBuilder,
)

import agimus_controller_mod.trajectories.saw_line_cartesian_space as traj


class SawLineCartesianSpace(TrajectoryBuilder):
    """Builder for saw-line trajectory/segment."""

    def from_params(self,
                    params: trajectory_parameters.Params,
                    nq: int,
                    ee_frame_name: str,
                    logger: RcutilsLogger) -> traj.SawLineCartesianSpace:
        """Build the full saw-line trajectory from ROS parameters."""
        weights = get_all_weights(params, nq, ee_frame_name)

        return traj.SawLineCartesianSpace(
            x=params.line_endpoints.x,
            transition_time=params.line_endpoints.time,
            w_mul=params.line_endpoints.w_mul,
            ee_frame_name=ee_frame_name,
            rotation_rpy=params.line_endpoints.rotation,
            weights=weights,
            tooth_length=params.saw.tooth_length,
            tooth_tip=params.saw.tooth_tip,
            info_logger=logger.info,
        )

    def from_goal(self,
                  goal: TrajectoryGoal,
                  nq: int) -> traj.SawLineSegmentCartesianSpace:
        """Build a single saw-line segment from an action goal."""
        weights = get_all_weights(goal, nq, goal.frame_name)
        segment = traj.SawLineSegmentCartesianSpace(
            goal.frame_name, weights, goal.s1, np.array(goal.v1))

        set_segment(goal, segment)
        return segment

    def to_goal(self, segment: traj.SawLineSegmentCartesianSpace,
                goal: TrajectoryGoal) -> TrajectoryGoal:
        set_goal(goal, segment)
        goal.trajectory_type = __name__.rpartition(".")[-1] + ':' + \
                               self.__class__.__name__

        goal.s1 = segment.tooth_length
        goal.v1 = list(segment.tooth_tip)

        return goal
