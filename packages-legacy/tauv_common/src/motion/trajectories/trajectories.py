# Trajectories.py
#
# This is the home for the Tartan AUV Trajectory profiles!
#
# Each profile extends the abstract base class (abc) Trajectory, and is used to
# describe a single controlled motion.
#
# Useful resource on trajectory planning:
# https://medium.com/mathworks/trajectory-planning-for-robot-manipulators-522404efb6f0
#
# Author: Tom Scherlis
#

import abc
from enum import Enum
from nav_msgs.msg import Path
from collections import Iterable


class TrajectoryStatus(Enum):
    PENDING = 0      # Trajectory has not been initialized yet
    EXECUTING = 1    # MPC controller is actively tracking this trajectory
    FINISHED = 3     # Trajectory has been completed (ie, all useful points have been published to controller)
    STABILIZED = 4   # AUV has settled to within some tolerance of the final goal location.


class Trajectory(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_points(self, request):
        pass

    @abc.abstractmethod
    def get_segment_duration(self):
        pass

    @abc.abstractmethod
    def start(self):
        pass

    @abc.abstractmethod
    def get_status(self, pose):
        pass

    @abc.abstractmethod
    def as_path(self):
        pass

    @abc.abstractclassmethod
    def get_target(self):
        pass