import time
import rospy
from spatialmath import SE3, SE3, SO3, SE2
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
from typing import Optional
from std_srvs.srv import SetBool
from tauv_msgs.msg import PingDetection
from math import atan2, cos, sin
from tauv_util.spatialmath import ros_vector3_to_r3
from trajectories.constant_acceleration_trajectory import ConstantAccelerationTrajectoryParams
import numpy as np

class DetectPingerStatus(TaskStatus):
    SUCCESS = 0
    TIMEOUT = 1
    CANCELLED = 2

@dataclass
class DetectPingerResult(TaskResult):
    status: DetectPingerStatus
    location: str = "octagon"

class DetectPinger(Task):

    def __init__(self, frequency: float, depth: float, course_t_torpedo: SE3, course_t_octagon: SE3, n_pings: int):
        super().__init__()

        self._frequency = frequency
        self._depth = depth

        self._course_t_torpedo = course_t_torpedo
        self._course_t_octagon = course_t_octagon

        self._n_pings = n_pings

        self._arm_srv: rospy.ServiceProxy = rospy.ServiceProxy('vehicle/thrusters/arm', SetBool)

    def run(self, resources: TaskResources) -> DetectPingerResult:
        timeout_time = rospy.Time.now() + rospy.Duration.from_sec(self._timeout)

        resources.motion.goto_relative_with_depth(SE2(0, 0, 0), self._depth)

        torpedo_votes = 0
        octagon_votes = 0

        for i in range(self._n_pings):
            detection = None

            try:
                self._arm_srv.call(False)
                detection = rospy.wait_for_message('vehicle/pinger_localizer/detection', PingDetection, timeout=3.0)
                self._arm_srv.call(True)
            except Exception:
                continue

            if detection is None:
                continue

            if abs(detection.frequency - self._frequency) > 500:
                continue

            rospy.loginfo(detection)

            odom_t_course = resources.transforms.get_a_to_b('kf/odom', 'kf/course')
            odom_t_torpedo = odom_t_course * self._course_t_torpedo
            odom_t_octagon = odom_t_course * self._course_t_octagon

            odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')

            direction = ros_vector3_to_r3(detection.direction)

            direction_yaw = atan2(direction[1], direction[0]) + odom_t_vehicle.rpy()[2]

            torpedo_yaw = atan2(odom_t_torpedo.t[1] - odom_t_vehicle.t[1], odom_t_torpedo.t[0] - odom_t_vehicle.t[0])
            octagon_yaw = atan2(odom_t_octagon.t[1] - odom_t_vehicle.t[1], odom_t_octagon.t[0] - odom_t_vehicle.t[0])

            # TODO: Make this the actual angle between two angles``

            if (direction_yaw - torpedo_yaw) < 0.1:
                torpedo_votes += 1
            elif (direction_yaw - octagon_yaw) < 0.1:
                octagon_votes += 1

            if self._check_cancel(resources): return DetectPingerResult(DetectPingerStatus.CANCELLED)

        if rospy.Time.now() > timeout_time:
            return DetectPingerResult(DetectPingerStatus.TIMEOUT)

        location = "octagon"
        if torpedo_votes > octagon_votes and torpedo_votes > 7:
            location = "torpedo"
        if octagon_votes > torpedo_votes and octagon_votes > 7:
            location = "octagon"

        return DetectPingerResult(DetectPingerStatus.SUCCESS, location=location)



    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()