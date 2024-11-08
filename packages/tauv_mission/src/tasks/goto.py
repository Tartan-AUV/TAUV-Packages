import time
import rospy
from dataclasses import dataclass
from spatialmath import SE3
from tasks.task import Task, TaskResources, TaskStatus, TaskResult


class GotoStatus(TaskStatus):
    SUCCESS = 0
    FAILURE = 1


@dataclass
class GotoResult(TaskResult):
    status: GotoStatus


class Goto(Task):

    def __init__(self, pose: SE3, in_course: bool = False, delay: float = 0):
        super().__init__()

        self._pose: SE3 = pose
        self._in_course = in_course
        self._delay = delay

    def run(self, resources: TaskResources) -> GotoResult:
        if self._in_course:
            pose = resources.transforms.get_a_to_b('kf/odom', 'kf/course') * self._pose
        else:
            pose = self._pose
        resources.motion.goto(pose, params=resources.motion.get_trajectory_params("rapid"))

        while True:
            if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                break

            if self._check_cancel(resources): return GotoResult(GotoStatus.FAILURE)

        time.sleep(self._delay)

        return GotoResult(GotoStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()