import rospy
import numpy as np
from math import pi
from spatialmath import SE3, SO3
from dataclasses import dataclass

from std_srvs.srv import SetBool

from tasks.task import Task, TaskResources, TaskStatus, TaskResult
import time
from tauv_util.spatialmath import flatten_se3


class BarrelRollStatus(TaskStatus):
    SUCCESS = 0
    CANCELLED = 1


@dataclass
class BarrelRollResult(TaskResult):
    status: BarrelRollStatus


class BarrelRoll(Task):
    def __init__(self, n_roll_waypoints=9):
        super().__init__()

        self._n_roll_waypoints = n_roll_waypoints

    def run(self, resources: TaskResources) -> BarrelRollResult:
        do_barrel_roll = rospy.ServiceProxy('gnc/do_barrel_roll', SetBool)
        do_barrel_roll()
        time.sleep(4)
        do_barrel_roll()
        time.sleep(4)

        return BarrelRollResult(status=BarrelRollStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()
