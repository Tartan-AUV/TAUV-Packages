from typing import Optional
from math import pi
from spatialmath import SE3, SO3, SE2
from missions.mission import Mission
from tasks.task import Task, TaskResult
from tasks import dive, goto, surface


class KFPrequal24(Mission):

    def __init__(self):
        depth = 2

        self._waypoint_i: int = 0
        self._waypoints: [SE3] = [
            SE3.Rt(R=SO3(), t=(0.5, 0, depth)),
            SE3.Rt(R=SO3(), t=(3, 0, depth)),
            SE3.Rt(R=SO3(), t=(12, 1, depth)),
            SE3.Rt(R=SO3.Rz(-pi / 4), t=(12, 1, depth)),
            SE3.Rt(R=SO3.Rz(-pi / 4), t=(13, 0, depth)),
            SE3.Rt(R=SO3.Rz(-3 * pi / 4), t=(13, 0, depth)),
            SE3.Rt(R=SO3.Rz(-3 * pi / 4), t=(12, -1, depth)),
            SE3.Rt(R=SO3.Rz(-pi), t=(12, -1, depth)),
            SE3.Rt(R=SO3.Rz(-pi), t=(3, 0, depth)),
            SE3.Rt(R=SO3.Rz(-pi), t=(1, 0, depth)),
        ]

    def entrypoint(self) -> Optional[Task]:
        self._waypoint_i = 0
        return dive.Dive(delay=20.0, x_offset=0.5, y_offset=0)

    def transition(self, task: Task, task_result: TaskResult) -> Optional[Task]:
        if self._waypoint_i < len(self._waypoints):
            task = goto.Goto(self._waypoints[self._waypoint_i], in_course=True)
            self._waypoint_i += 1
            return task
        elif self._waypoint_i == len(self._waypoints):
            task = surface.Surface()
            self._waypoint_i += 1
            return task
        else:
            return None
