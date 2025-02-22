import time
import rospy
from spatialmath import SE2, SE3, SO3
import numpy as np
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult


class ScanTranslateStatus(TaskStatus):
    SUCCESS = 0
    FAILURE = 1


@dataclass
class ScanTranslateResult(TaskResult):
    status: ScanTranslateStatus


class ScanTranslate(Task):

    def __init__(self, course_t_start: SE3, points: [(float, float)]):
        super().__init__()

        self._course_t_start = course_t_start
        self._points = points

    def run(self, resources: TaskResources) -> ScanTranslateResult:

        odom_t_course = resources.transforms.get_a_to_b('kf/odom', 'kf/course')
        odom_t_start = odom_t_course * self._course_t_start

        for (y, z) in self._points:
            start_t_vehicle_goal = SE3.Rt(SO3(), (0, y, z))
            odom_t_vehicle_goal = odom_t_start * start_t_vehicle_goal

            resources.motion.goto(odom_t_vehicle_goal)

            while True:
                if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                    break

                if self._check_cancel(resources): return ScanTranslateResult(ScanTranslateStatus.FAILURE)

        return ScanTranslateResult(ScanTranslateStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()