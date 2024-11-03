from typing import Optional
from spatialmath import SE3, SO3, SE2
from missions.mission import Mission
from tasks.task import Task, TaskResult
from tasks import dive, goto, gate, detect_pinger, buoy_search, gate_dead_reckon, surface
from enum import IntEnum


class KFTransdec23State(IntEnum):
    UNKNOWN = 0
    DIVE = 1
    GOTO_GATE = 2
    GOTO_BUOY = 3
    GOTO_OCTAGON = 4
    SURFACE = 5

class KFTransdec23(Mission):

    def __init__(self):
        self._state = KFTransdec23State.UNKNOWN

        self._dive_y_offset = 1

        self._buoy_xy_steps = [
            (0, -2),
            (1, -1.5),
            (0, -1),
            (1, -0.5),
            (0, 0),
            (1, 0.5),
            (0, 1),
            (1, 1.5),
            (0, 2),
        ]
        self._buoy_z_steps = [
            0,
            0.75
        ]

        self._course_t_start: SE3 = SE3.Rt(SO3(), (2, 3, 1.5))
        self._course_t_gate: SE3 = SE3.Rt(SO3.Rz(0.2), (6.75, 4.4, 1.5))
        self._course_t_buoy: SE3 = SE3.Rt(SO3(), (14.7, -22, 1.5))
        self._course_t_octagon: SE3 = SE3.Rt(SO3(), (32, -22, 1.5))

        self._pinger_frequency: int = 30000

    def entrypoint(self) -> Optional[Task]:
        self._state = KFTransdec23State.DIVE
        return dive.Dive(delay=20.0, y_offset=self._dive_y_offset)

    def transition(self, task: Task, task_result: TaskResult) -> Optional[Task]:
        if self._state == KFTransdec23State.DIVE:
        #     self._state = KFTransdec23State.GOTO_GATE
        #     return goto.Goto(self._course_t_gate, in_course=True, delay=10.0)
        # elif self._state == KFTransdec23State.GOTO_GATE:
            self._state = KFTransdec23State.GOTO_BUOY
            return goto.Goto(self._course_t_buoy, in_course=True, delay=10.0)
        elif self._state == KFTransdec23State.GOTO_BUOY:
            self._state = KFTransdec23State.GOTO_OCTAGON
            return goto.Goto(self._course_t_octagon, in_course=True, delay=10.0)
        elif self._state == KFTransdec23State.GOTO_OCTAGON:
            self._state = KFTransdec23State.SURFACE
            return surface.Surface()
        else:
            return None

        # TODO: ADD PAUSE