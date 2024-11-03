import time
import rospy
from dataclasses import dataclass
from spatialmath import SE3, SE2
from tasks.task import Task, TaskResources, TaskStatus, TaskResult


class SurfaceStatus(TaskStatus):
    SUCCESS = 0
    FAILURE = 1


@dataclass
class SurfaceResult(TaskResult):
    status: SurfaceStatus


class Surface(Task):

    def __init__(self):
        super().__init__()

    def run(self, resources: TaskResources) -> SurfaceResult:
        resources.motion.goto_relative_with_depth(SE2(0, 0, 0), 0)

        while True:
            if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                break

            if self._check_cancel(resources): return SurfaceResult(SurfaceStatus.FAILURE)

        try:
            resources.motion.arm(False)
        except Exception:
            pass

        return SurfaceResult(SurfaceStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()