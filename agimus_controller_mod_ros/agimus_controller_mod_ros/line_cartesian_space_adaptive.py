from agimus_demos_common.trajectories.line_segment_cartesian_space_adaptive import \
    LineSegmentCartesianSpaceAdaptive

from agimus_demos_common.trajectories.line_cartesian_space import \
    LineCartesianSpace


class LineCartesianSpaceAdaptive(LineCartesianSpace):

    def __init__(self, dt: float, **kwargs):
        super().__init__(**kwargs)

        self.segment = LineSegmentCartesianSpaceAdaptive(self.ee_frame_name, dt)
        self.segment.logger = self.logger
