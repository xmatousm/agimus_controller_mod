from rclpy.impl.rcutils_logger import RcutilsLogger

from .trajectory_builder import (
    get_all_weights,
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
        trajectory = traj.LineSegmentCartesianSpace(goal.frame_name, weights)
        trajectory.goal_weight_boost = goal.goal_weight_boost
        trajectory.goal_tolerance_boost = goal.goal_tolerance_boost
        return trajectory
