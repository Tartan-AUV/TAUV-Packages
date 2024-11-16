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
    GATE = 3
    GOTO_BUOY = 4
    BUOY = 5
    CLEAR_BUOY = 6
    GOTO_OCTAGON = 7
    SURFACE = 8

class KFTransdec23(Mission):

    def __init__(self):
        self._state = KFTransdec23State.UNKNOWN

        self._dive_y_offset = -1

        self._buoy_xy_steps = [
            (0, -1),
            (1, -1),
            (0, -1),
            (0, 0),
            (1, 0),
            (0, 0),
            (0, 1),
            (1, 1),
            (0, 1),
        ]
        self._buoy_z_steps = [
            0,
            0.5
        ]

        self._course_t_start: SE3 = SE3.Rt(SO3(), (2, -3, 1.5))
        self._course_t_gate: SE3 = SE3.Rt(SO3.Rz(0.0), (7.5, -3.4, 1.5))
        self._course_t_buoy: SE3 = SE3.Rt(SO3(), (14.7, -9.8, 2.5))
        self._course_t_clear_buoy: SE3 = SE3.Rt(SO3(), (12, -7.3, 1.5))
        self._course_t_octagon: SE3 = SE3.Rt(SO3(), (32, -22, 1.5))

        self._pinger_frequency: int = 30000

    def entrypoint(self) -> Optional[Task]:
        self._state = KFTransdec23State.DIVE
        return dive.Dive(delay=20.0, y_offset=self._dive_y_offset)

    def transition(self, task: Task, task_result: TaskResult) -> Optional[Task]:
        if self._state == KFTransdec23State.DIVE:
            self._state = KFTransdec23State.GOTO_GATE
            return goto.Goto(self._course_t_start, in_course=True)
        elif self._state == KFTransdec23State.GOTO_GATE:
            self._state = KFTransdec23State.GATE
            return gate_dead_reckon.Gate(course_t_gate=self._course_t_gate)
        elif self._state == KFTransdec23State.GATE:
        # self._state = KFTransdec23State.GOTO_BUOY
        #     return goto.Goto(self._course_t_buoy, in_course=True)
        # elif self._state == KFTransdec23State.GOTO_BUOY:
        #     self._state = KFTransdec23State.BUOY
        #     return buoy_search.BuoySearch(course_t_start
        #                                   =self._course_t_buoy, xy_steps=self._buoy_xy_steps, z_steps=self._buoy_z_steps)
        # elif self._state == KFTransdec23State.BUOY:
        #     self._state = KFTransdec23State.CLEAR_BUOY
        #     return goto.Goto(self._course_t_clear_buoy, in_course=True)
        # elif self._state == KFTransdec23State.CLEAR_BUOY:
            self._state = KFTransdec23State.GOTO_OCTAGON
            return goto.Goto(self._course_t_octagon, in_course=True)
        elif self._state == KFTransdec23State.GOTO_OCTAGON:
            self._state = KFTransdec23State.SURFACE
            return surface.Surface()
        else:
            return None