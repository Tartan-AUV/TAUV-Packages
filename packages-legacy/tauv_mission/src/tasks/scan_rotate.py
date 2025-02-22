import time
import rospy
import numpy as np
from spatialmath import SE2, SE3, SO3
from math import pi
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult


class ScanRotateStatus(TaskStatus):
    SUCCESS = 0
    FAILURE = 1


@dataclass
class ScanRotateResult(TaskResult):
    status: ScanRotateStatus


class ScanRotate(Task):

    def __init__(self):
        super().__init__()

    def run(self, resources: TaskResources) -> ScanRotateResult:
        odom_t_vehicle_initial = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')

        for i in range(8):
            theta = i * (2 * pi) / 8

            rospy.loginfo(theta)

            resources.motion.goto(odom_t_vehicle_initial * SE3.Rz(theta))

            while True:
                if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                    break

                if self._check_cancel(resources): return ScanRotateResult(ScanRotateStatus.FAILURE)

            time.sleep(5.0)

        return ScanRotateResult(ScanRotateStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()