import rospy
import numpy as np
from math import pi
from spatialmath import SE3, SO3
from dataclasses import dataclass

from std_srvs.srv import SetBool

from tasks.task import Task, TaskResources, TaskStatus, TaskResult
import time
from tauv_util.spatialmath import flatten_se3


class DropMarkerStatus(TaskStatus):
    SUCCESS = 0
    CANCELLED = 1


@dataclass
class DropMarkerResult(TaskResult):
    status: DropMarkerStatus


class DropMarker(Task):
    def __init__(self):
        super().__init__()

    def run(self, resources: TaskResources):
        time.sleep(2)
        resources.actuators.drop_marker(1)
        time.sleep(1)
        resources.actuators.drop_marker(0)
        time.sleep(2)

        return DropMarkerResult(DropMarkerStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()
