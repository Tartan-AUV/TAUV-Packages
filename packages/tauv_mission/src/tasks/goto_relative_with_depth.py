import time
import rospy
from dataclasses import dataclass
from spatialmath import SE3, SE2
from tasks.task import Task, TaskResources, TaskStatus, TaskResult


class GotoRelativeWithDepthStatus(TaskStatus):
    SUCCESS = 0
    FAILURE = 1


@dataclass
class GotoRelativeWithDepthResult(TaskResult):
    status: GotoRelativeWithDepthStatus


class GotoRelativeWithDepth(Task):

    def __init__(self, pose: SE2, depth: float):
        super().__init__()

        self._pose: SE2 = pose
        self._depth: float = depth

    def run(self, resources: TaskResources) -> GotoRelativeWithDepthResult:
        resources.motion.goto_relative_with_depth(self._pose, self._depth)

        while True:
            if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                break

            if self._check_cancel(resources): return GotoRelativeWithDepthResult(GotoRelativeWithDepthStatus.FAILURE)

        return GotoRelativeWithDepthResult(GotoRelativeWithDepthStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()