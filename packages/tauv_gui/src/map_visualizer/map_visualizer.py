import rospy
from threading import Lock
from enum import Enum
from tauv_msgs.msg import FeatureDetection, FeatureDetections
from tauv_msgs.srv import MapFind, MapFindRequest
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
from typing import Any
from spatialmath import SE3, SO3
from tauv_util.spatialmath import ros_point_to_r3, se3_to_ros_pose


class Source(Enum):
    DETECTION = "detection"
    MAP = "map"


class MapVisualizer:

    def __init__(self):
        self._lock: Lock = Lock()
        with self._lock:
            self._load_config()

            self._n_detection: {str: int} = {}
            self._n_map: {str: int} = {}

            self._detection_sub: rospy.Subscriber =\
                rospy.Subscriber("global_map/feature_detections", FeatureDetections, self._handle_detections)

            self._marker_pub: rospy.Publisher =\
                rospy.Publisher("global_map/visualization_marker_array", MarkerArray, queue_size=10)

            self._map_timer: rospy.Timer = rospy.Timer(rospy.Duration.from_sec(self._map_publish_rate), self._handle_map)

            self._find_srv: rospy.ServiceProxy = rospy.ServiceProxy("global_map/find", MapFind)
            rospy.wait_for_service("global_map/find")

    def start(self):
        rospy.spin()

    def _handle_detections(self, msg: FeatureDetections):
        with self._lock:
            markers = MarkerArray()

            for detection in msg.detections:
                if detection.tag not in self._n_detection:
                    self._n_detection[detection.tag] = 0
                try:
                    marker = self._get_marker(detection, source=Source.DETECTION)
                    marker.header.frame_id = f"{self._tf_namespace}/odom"
                    marker.ns = f"{detection.tag}-detection"
                    marker.id = self._n_detection[detection.tag]
                    markers.markers.append(marker)
                    self._n_detection[detection.tag] += 1
                except ValueError:
                    continue

            self._marker_pub.publish(markers)

    def _handle_map(self, _):
        with self._lock:
            markers = MarkerArray()

            for tag in self._tags:
                req = MapFindRequest()
                req.tag = tag

                res = self._find_srv.call(req)

                if not res.success:
                    continue

                for detection_i, detection in enumerate(res.detections):
                    try:
                        marker = self._get_marker(detection, source=Source.MAP)
                        marker.header.frame_id = f"{self._tf_namespace}/odom"
                        marker.ns = f"{tag}-map"
                        marker.id = detection_i
                        markers.markers.append(marker)
                    except ValueError:
                        continue

                    # TODO: Add text?

                if tag in self._n_map and len(res.detections) < self._n_map[tag]:
                    for delete_i in range(len(res.detections), self._n_map[tag]):
                        marker = Marker()
                        marker.action = marker.DELETE
                        marker.ns = f"{tag}-map"
                        marker.id = delete_i
                        markers.markers.append(marker)

                self._n_map[tag] = len(res.detections)

            self._marker_pub.publish(markers)

    def _get_marker(self, detection: FeatureDetection, source: Source = Source.MAP) -> Marker:
        marker_type = self._get_marker_type(detection.tag, source)

        if marker_type == "frame":
            return self._get_frame_marker(detection, source)
        elif marker_type == "arrow":
            return self._get_arrow_marker(detection, source)
        else:
            raise ValueError(f"unknown visualization type {marker_type}")

    def _get_frame_marker(self, detection: FeatureDetection, source: Source) -> Marker:
        scale = self._get_marker_param(detection.tag, "frame", source, "scale")
        lifetime = self._get_marker_param(detection.tag, "frame", source, "lifetime")

        marker = Marker()

        marker.type = marker.LINE_LIST
        marker.points = [
            Point(0, 0, 0),
            Point(scale, 0, 0),
            Point(0, 0, 0),
            Point(0, scale, 0),
            Point(0, 0, 0),
            Point(0, 0, scale),
        ]
        marker.colors = [
            ColorRGBA(1.0, 0.0, 0.0, 1.0),
            ColorRGBA(1.0, 0.0, 0.0, 1.0),
            ColorRGBA(0.0, 1.0, 0.0, 1.0),
            ColorRGBA(0.0, 1.0, 0.0, 1.0),
            ColorRGBA(0.0, 0.0, 1.0, 1.0),
            ColorRGBA(0.0, 0.0, 1.0, 1.0),
        ]
        marker.scale = Vector3(0.01, 0, 0)

        marker.action = marker.ADD
        marker.frame_locked = True
        marker.lifetime = rospy.Duration.from_sec(lifetime)

        pose = SE3.Rt(SO3.RPY(ros_point_to_r3(detection.orientation)), ros_point_to_r3(detection.position))
        marker.pose = se3_to_ros_pose(pose)

        return marker

    def _get_arrow_marker(self, detection: FeatureDetection, source: Source):
        scale = self._get_marker_param(detection.tag, "arrow", source, "scale")
        color = tuple(self._get_marker_param(detection.tag, "arrow", source, "color"))
        opacity = self._get_marker_param(detection.tag, "arrow", source, "opacity")
        lifetime = self._get_marker_param(detection.tag, "arrow", source, "lifetime")
        axis = self._get_marker_param(detection.tag, "arrow", source, "axis")

        marker = Marker()
        marker.type = marker.ARROW
        marker.scale = Vector3(scale, 0.1 * scale, 0.1 * scale)
        marker.color = ColorRGBA(color[0], color[1], color[2], opacity)

        marker.action = marker.ADD
        marker.frame_locked = True
        marker.lifetime = rospy.Duration.from_sec(lifetime)

        pose = SE3.Rt(SO3.RPY(ros_point_to_r3(detection.orientation)), ros_point_to_r3(detection.position))

        if axis not in ["+x", "-x", "+z", "-z"]:
            raise ValueError(f"unknown axis {axis}")
        pose_adjustment = SE3(SO3.TwoVectors(x=axis, y="+y"))

        pose = pose * pose_adjustment

        marker.pose = se3_to_ros_pose(pose)

        return marker

    def _get_marker_type(self, tag: str, source: Source) -> str:
        if tag in self._marker_params and source.value in self._marker_params[tag] and "type" in self._marker_params[tag][source.value]:
            return self._marker_params[tag][source.value]["type"]

        return self._marker_params["default"][source.value]["type"]

    def _get_marker_param(self, tag: str, marker_type: str, source: Source, param: str) -> Any:
        if tag in self._marker_params and source.value in self._marker_params[tag]\
                and param in self._marker_params[tag][source.value]:
            return self._marker_params[tag][source.value][param]

        if param in self._marker_params["default"][source.value]:
            return self._marker_params["default"][source.value][param]

        return self._marker_types[marker_type][param]

    def _load_config(self):
        self._tf_namespace: str = rospy.get_param("tf_namespace")
        self._map_publish_rate: float = rospy.get_param("~map_publish_rate")
        self._tags: [str] = rospy.get_param("~tags")
        self._marker_types: {str: Any} = rospy.get_param("~marker_types")
        self._marker_params: {str: Any} = rospy.get_param("~markers")


def main():
    rospy.init_node('map_visualizer')
    n = MapVisualizer()
    n.start()
