import rospy
import numpy as np
from spatialmath import SE3, SO3, SE2
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
import time
from tauv_util.spatialmath import flatten_se3
from math import atan2


class BuoySearchStatus(TaskStatus):
    SUCCESS = 0
    CANCELLED = 1
    TIMEOUT = 2

@dataclass
class BuoySearchResult(TaskResult):
    status: BuoySearchStatus


class BuoySearch(Task):

    def __init__(self, course_t_start: SE3, xy_steps: [(int, int)], z_steps: [int]):
        super().__init__()

        self._course_t_start = course_t_start
        self._xy_steps = xy_steps
        self._z_steps = z_steps

    def run(self, resources: TaskResources) -> BuoySearchResult:
        timeout_time = rospy.Time.now() + rospy.Duration.from_sec(10000)

        resources.motion.cancel()

        odom_t_course = resources.transforms.get_a_to_b('kf/odom', 'kf/course')

        for z in self._z_steps:
            for (x, y) in self._xy_steps:
                start_t_goal = SE3.Rt(SO3(), (x, y, z))

                course_t_goal = self._course_t_start * start_t_goal
                odom_t_goal = odom_t_course * course_t_goal

                resources.motion.goto(odom_t_goal, params=resources.motion.get_trajectory_params("rapid"))

                if self._spin_cancel(resources, lambda: resources.motion.wait_until_complete(
                        timeout=rospy.Duration.from_sec(0.1)), timeout_time):
                    return BuoySearchResult(status=BuoySearchStatus.TIMEOUT)

        resources.motion.goto_relative_with_depth(SE2(0, 0, 0), 1.5)

        if self._spin_cancel(resources, lambda: resources.motion.wait_until_complete(
                timeout=rospy.Duration.from_sec(0.1)), timeout_time):
            return BuoySearchResult(status=BuoySearchStatus.TIMEOUT)

        return BuoySearchResult(status=BuoySearchStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()
