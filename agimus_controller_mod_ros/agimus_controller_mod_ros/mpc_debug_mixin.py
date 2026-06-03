from typing import Optional, Callable
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import QoSProfile, ReliabilityPolicy

from builtin_interfaces.msg import Duration as DurationMsg
from visualization_msgs.msg import MarkerArray, Marker
from agimus_controller_ros.ros_utils import (
    se3_to_pose_msg,
)
from builtin_interfaces.msg import Time as MsgTime
from geometry_msgs.msg import Pose, Point

from agimus_controller.trajectory import TrajectoryBuffer
from agimus_controller.mpc_data import MPCDebugData
import pinocchio
import numpy as np


def create_markers(frame_id: str,
                   namespace: str, count: int,
                   marker_size: float,
                   rgba0, rgba1,
                   time_stamp: Optional[MsgTime] = None,
                   marker_type=Marker.SPHERE,
                   **kwargs
                   ) -> list[Marker]:
    markers = []
    for i in range(count):
        marker = Marker()
        marker.header.frame_id = frame_id
        if time_stamp is not None:
            marker.header.stamp = time_stamp
        marker.ns = namespace
        marker.id = i

        marker.type = marker_type

        marker.action = Marker.ADD

        marker.scale.x = marker_size
        marker.scale.y = marker_size
        marker.scale.z = marker_size

        # Interpolate from rgba0 to rgba1
        r = i / (count - 1)
        marker.color.r = rgba0[0] + r * (rgba1[0] - rgba0[0])
        marker.color.g = rgba0[1] + r * (rgba1[1] - rgba0[1])
        marker.color.b = rgba0[2] + r * (rgba1[2] - rgba0[2])
        marker.color.a = rgba0[3] + r * (rgba1[3] - rgba0[3])

        marker.lifetime = DurationMsg(sec=1)

        markers.append(marker)
    return markers


def set_marker_ortho_line(m: Marker, x0, vec, scale: float):
    ortho1 = np.cross(vec, np.array([1.0, 0.0, 0.0]))
    ortho2 = np.cross(vec, np.array([0.0, 0.1, 0.0]))

    if np.linalg.norm(ortho1) > 1e-7:
        ortho = ortho1 / np.linalg.norm(ortho1)
    elif np.linalg.norm(ortho2) > 1e-7:
        ortho = ortho2 / np.linalg.norm(ortho2)
    else:
        ortho = np.array([0.0, 0.0, 1.0])

    x1 = x0 + ortho * scale
    m.points = [Point(x=x0[0], y=x0[1], z=x0[2]),
                Point(x=x1[0], y=x1[1], z=x1[2])]


class MpcDebugMarkers:
    def __init__(self, node: Node, frame_name: str, parent_frame_name: str,
                 horizon_size: int, horizon_size_full: int,
                 marker_size: float = 0.01):
        self.node = node
        self._frame_name = frame_name
        self._parent_frame_name = parent_frame_name
        self.marker_size = marker_size
        self._horizon_size = horizon_size
        self._horizon_size_full = horizon_size_full

        self._rmodel = None
        self._rdata = None
        self._frame_id = None
        self._initialized = False

        self.spec = {
            "ref_state": {
                "namespace": "states_references",
                "topic": "mpc_states_reference_markers",
                "count": horizon_size_full + 1,
                "marker_type": Marker.SPHERE,
                "marker_size": marker_size/ 10,
                "rgba0": [0.0, 0.0, 1.0, 1.0],
                "rgba1": [1.0, 0.0, 1.0, 0.2],
            },
            "ref_pose": {
                "namespace": "states_references_pose",
                "topic": "mpc_states_reference_markers_pose",
                "count": horizon_size_full + 1,
                "marker_type": Marker.SPHERE,
                "marker_size": marker_size / 10,
                "rgba0": [0.0, 1.0, 0.0, 1.0],
                "rgba1": [0.0, 0.5, 1.0, 0.2],
            },
            "cur_input": {
                "namespace": "pose_input",
                "topic": "mpc_current_reference_markers_pose",
                "count": horizon_size_full + 1,
                "marker_type": Marker.LINE_STRIP,
                "marker_size": marker_size / 10,
                "rgba0": [1.0, 1.0, 0.0, 1.0],
                "rgba1": [1.0, 1.0, 0.0, 0.2],
            },
            "pred": {
                "namespace": "states_predictions",
                "topic": "mpc_states_prediction_markers",
                "count": horizon_size + 1,
                "marker_type": Marker.SPHERE,
                "marker_size": marker_size / 10,
                "rgba0": [1.0, 0.0, 0.0, 1.0],
                "rgba1": [0.5, 1.0, 0.0, 0.2],
            },
        }

        # create empty marker arrays (initialized later, so a client can change
        # the spec) and corresponding publishers
        self._marker_array: dict[str, MarkerArray] = {}
        self._markers: dict[str, list[Marker]] = {}
        self._publisher: dict[str, Publisher] = {}

        for key in self.spec:
            self._marker_array[key] = MarkerArray()
            self._markers[key] = []
            self._publisher[key] = node.create_publisher(
                MarkerArray,
                self.spec[key]["topic"],
                qos_profile=QoSProfile(
                    depth=10,
                    reliability=ReliabilityPolicy.RELIABLE,
                ),
            )

    def _publish(self, key: str):
        """Publish the markers determined by the key using the corresponding
         publisher."""
        self._publisher[key].publish(self._marker_array[key])

    def initialize(self, rmodel):
        # init robot model
        self._rmodel = rmodel
        assert self._rmodel.existFrame(self._frame_name), \
            f"Frame {self._frame_name} could not be found."
        self._rdata = self._rmodel.createData()
        self._frame_id = self._rmodel.getFrameId(self._frame_name)
        self._initialized = True

        # prepare all marker arrays
        for key in self.spec:
            self._markers[key] = create_markers(
                    frame_id=self._parent_frame_name, **self.spec[key])
            self._marker_array[key] = MarkerArray(markers=self._markers[key])

    def mpc_input_publish(self, references: TrajectoryBuffer):
        if not self._initialized:
            return

        for i in range(self._horizon_size + 1):
            m: Marker = self._markers['cur_input'][i]
            if len(references) > i:
                point = references[i].point
                pose = list(point.end_effector_poses.values())[0]
                #m.pose = se3_to_pose_msg(pose)
                # TODO use the correct ee frame from dict

                # line in an orthogonal direction of the motion
                if i > 0:
                    vec = pose.translation - pose_old.translation
                    set_marker_ortho_line(m, pose.translation, vec,
                                          m.scale.x * 20)
                    if i == 1:  # set the first marker using vec
                        set_marker_ortho_line(
                            self._markers['cur_input'][0],
                            pose_old.translation, vec, m.scale.x * 20)

                pose_old = pose

            else:
                m.pose = Pose()

        self._publish('cur_input')

    def mpc_references_publish(self, references: TrajectoryBuffer):
        if not self._initialized:
            return

        for i in range(self._horizon_size + 1):
            ms: Marker = self._markers['ref_state'][i]
            mp: Marker = self._markers['ref_pose'][i]
            if len(references) > i:
                point = references[i].point
                state = point.robot_configuration
                pose = list(point.end_effector_poses.values())[0]
                # TODO use the correct ee frame from dict
                mp.pose = se3_to_pose_msg(pose)

                pinocchio.forwardKinematics(self._rmodel, self._rdata,
                                            np.asarray(state))
                se3 = pinocchio.updateFramePlacement(
                    self._rmodel, self._rdata, self._frame_id)
                ms.pose = se3_to_pose_msg(se3)

            else:
                mp.pose = Pose()
                ms.pose = Pose()

        self._publish('ref_state')
        self._publish('ref_pose')

    def mpc_debug_data_markers_publish(self, mpc_debug_data: MPCDebugData):
        if not self._initialized:
            return
        nq = self._rmodel.nq

        states = mpc_debug_data.ocp.result.states

        assert len(states) == len(self._markers['pred']), f"{len(states)} != {len(self._markers['pred'])}"

        for state, marker in zip(states, self._markers['pred']):
            pinocchio.forwardKinematics(self._rmodel, self._rdata, state[:nq])
            se3 = pinocchio.updateFramePlacement(self._rmodel, self._rdata,
                                                 self._frame_id)
            marker.pose = se3_to_pose_msg(se3)

        self._publish('pred')
