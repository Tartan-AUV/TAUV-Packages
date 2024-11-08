
import abc
from enum import Enum
from motion.trajectories.trajectories import TrajectoryStatus
from nav_msgs.msg import Path
from collections import Iterable
from tauv_msgs.srv import GetTrajectory
import typing
import numpy as np
from geometry_msgs.msg import Pose, Twist, Vector3, Quaternion, PoseStamped
from scipy.spatial.transform import Rotation
import rospy
from tauv_util.types import tl, tm

class HoldPos(object):
    def __init__(self, pos, heading, threshold_lin=0.1, threshold_h=0.1):
        self.pose = Pose()
        self.pose.position = Vector3(pos[0], pos[1], pos[2])
        self.pose.orientation = tm(Rotation.from_euler('ZYX', [heading,0,0]).as_quat(), Quaternion)
        self.threshold_lin = threshold_lin
        self.threshold_ang = threshold_h
        self.final_pos = pos
        self.final_h = heading

        self.status = TrajectoryStatus.PENDING

    def get_points(self, request: GetTrajectory._request_class) -> GetTrajectory._response_class:
        res = GetTrajectory._response_class()
        res.auto_twists = False
        res.poses = [self.pose] * request.len
        res.twists = [Twist()] * request.len
        res.success = True
        return res

    def get_segment_duration(self):
        raise NotImplementedError

    def start(self):
        self.status = TrajectoryStatus.EXECUTING

    def get_status(self, pose):
        R: Rotation = Rotation.from_quat(tl(pose.orientation))

        if np.linalg.norm(np.array(self.final_pos) - np.array(tl(pose.position))) <= self.threshold_lin and \
                abs(self.final_h - R.as_euler('ZYX')[0]) <= self.threshold_ang:
            return TrajectoryStatus.STABILIZED
        else:
            return TrajectoryStatus.FINISHED

    def as_path(self):
        path = Path()
        path.header.frame_id = 'odom_ned'
        path.header.stamp = rospy.Time.now()
        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.pose = self.pose
        path.poses = [ps]
        return path

    @abc.abstractclassmethod
    def get_target(self):
        pass