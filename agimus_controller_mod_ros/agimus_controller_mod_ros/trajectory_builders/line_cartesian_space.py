from rclpy.impl.rcutils_logger import RcutilsLogger

from .trajectory_builder import (
    get_all_weights,
    set_segment,
    set_goal,
    trajectory_parameters,
    TrajectoryGoal,
    TrajectoryBuilder,
)

import agimus_controller_mod.trajectories.line_cartesian_space as traj

class LineCartesianSpace(TrajectoryBuilder):
    """Builder for LineCartesianSpace trajectory/segment."""

    def from_params(self,
                    params: trajectory_parameters.Params,
                    nq: int,
                    ee_frame_name: str,
                    logger: RcutilsLogger) -> traj.LineCartesianSpace:
        """Build the full waypoint trajectory from ROS parameters."""
        weights = get_all_weights(params, nq, ee_frame_name)

        return traj.LineCartesianSpace(
            x=params.line_endpoints.x,
            transition_time=params.line_endpoints.time,
            w_mul=params.line_endpoints.w_mul,
            ee_frame_name=ee_frame_name,
            rotation_rpy=params.line_endpoints.rotation,
            weights=weights,
            goal_tolerance=params.line_endpoints.goal_tolerance,
            goal_tolerance_boost=params.line_endpoints.goal_tolerance_boost,
            goal_weight_boost=params.line_endpoints.goal_weight_boost,
            info_logger=logger.info,
        )

    def from_goal(self,
                  goal: TrajectoryGoal,
                  nq: int) -> traj.LineSegmentCartesianSpace:
        """Build a single line-segment from an action goal."""
        weights = get_all_weights(goal, nq, goal.frame_name)

        tol = goal.goal_tolerance if goal.goal_tolerance > 0.0 else None
        segment = traj.LineSegmentCartesianSpace(
            goal.frame_name,
            weights,
            goal_tolerance=tol,
            goal_tolerance_boost=goal.goal_tolerance_boost,
            goal_weight_boost=goal.goal_weight_boost)

        set_segment(goal, segment)
        return segment

    def to_goal(self, segment: traj.LineSegmentCartesianSpace,
                goal: TrajectoryGoal) -> TrajectoryGoal:

        set_goal(goal, segment)
        goal.trajectory_type = __name__.rpartition(".")[-1] + ':' + \
                               self.__class__.__name__

        goal.goal_tolerance = segment.goal_tolerance if segment.goal_tolerance is not None else -1.0
        goal.goal_weight_boost = segment.goal_weight_boost
        goal.goal_tolerance_boost = segment.goal_tolerance_boost

        return goal
