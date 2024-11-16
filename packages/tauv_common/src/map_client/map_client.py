import rospy
from dataclasses import dataclass
from typing import Optional, List
from spatialmath import SE3, SO3
from spatialmath.base.types import R3
import numpy as np

from tauv_msgs.srv import MapFind, MapFindRequest, MapFindResponse
from tauv_msgs.srv import MapFindClosest, MapFindClosestRequest, MapFindClosestResponse
from tauv_msgs.srv import MapFindOne, MapFindOneRequest, MapFindOneResponse
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse

from tauv_util.spatialmath import r3_to_ros_point, ros_point_to_r3

@dataclass
class MapDetection:
    tag: str
    pose: SE3


class MapClient:

    def __init__(self):
        self._find_srv: rospy.ServiceProxy = rospy.ServiceProxy('global_map/find', MapFind)
        self._find_one_srv: rospy.ServiceProxy = rospy.ServiceProxy('global_map/find_one', MapFindOne)
        self._find_closest_srv: rospy.ServiceProxy = rospy.ServiceProxy('global_map/find_closest', MapFindClosest)
        self._reset_srv: rospy.ServiceProxy = rospy.ServiceProxy('global_map/reset', Trigger)

    def find(self, tag: str) -> Optional[List[MapDetection]]:
        req = MapFindRequest()
        req.tag = tag

        try:
          res: MapFindResponse = self._find_srv(req)
        except Exception as e:
            rospy.logerr(f'received exception from global_map/find: {e}')
            return None

        if not res.success:
            rospy.logerr(f'received error from global_map/find: {res.message}')
            return None

        detections = [
            MapDetection(
                tag,
                SE3.Rt(SO3.RPY(ros_point_to_r3(detection.orientation), order="zyx"), ros_point_to_r3(detection.position)),
            )
            for detection in res.detections
        ]

        return detections

    def find_one(self, tag: str) -> Optional[MapDetection]:
        req = MapFindOneRequest()
        req.tag = tag

        try:
            res: MapFindOneResponse = self._find_one_srv(req)
        except Exception as e:
            rospy.logerr(f'received exception from global_map/find_one: {e}')
            return None

        if not res.success:
            rospy.logerr('freceived error from global_map/find_one: {res.message}')
            return None

        detection = MapDetection(
            tag,
            SE3.Rt(SO3.RPY(ros_point_to_r3(res.detection.orientation), order="zyx"), ros_point_to_r3(res.detection.position)),
        )

        return detection

    def find_closest(self, tag: str, position: R3) -> Optional[MapDetection]:
        if np.any(np.isnan(position)):
            return None

        req = MapFindClosestRequest()
        req.tag = tag
        req.point = r3_to_ros_point(position)

        try:
            res: MapFindClosestResponse = self._find_closest_srv(req)
        except Exception as e:
            rospy.logerr(f'received exception from global_map/find_closest: {e}')
            return None

        if not res.success:
            rospy.logerr(f'received error from global_map/find_closest')
            return None

        detection = MapDetection(
            tag,
            SE3.Rt(SO3.RPY(ros_point_to_r3(res.detection.orientation), order="zyx"), ros_point_to_r3(res.detection.position)),
        )

        return detection

    def reset(self) -> bool:
        req = TriggerRequest()
        res: TriggerResponse = self._reset_srv(req)

        if not res.success:
            rospy.logerr(f'received error from global_map/reset: {res.message}')
            return False

        return True

