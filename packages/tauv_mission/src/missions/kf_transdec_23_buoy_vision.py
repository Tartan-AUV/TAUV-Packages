from typing import Optional
from spatialmath import SE3, SO3, SE2
from missions.mission import Mission
from tasks.task import Task, TaskResult
from tasks import dive, goto, gate, detect_pinger, buoy_search, gate_dead_reckon, surface, hit_buoy, scan_translate


from enum import IntEnum


class KFTransdec23State(IntEnum):
    UNKNOWN = 0
    DIVE = 1
    GOTO_GATE = 2
    GATE = 3
    GOTO_BUOY = 4
    BUOY_SCAN = 5
    BUOY_1 = 6
    BUOY_2 = 7
    CLEAR_BUOY = 8
    GOTO_OCTAGON = 9
    SURFACE = 10

class KFTransdec23(Mission):

    def __init__(self):
        self._state = KFTransdec23State.UNKNOWN

        self._dive_y_offset = 1

        self._goto_octagon_depth = 1

        self._buoy_scan_points = [
            (-2, 0),
            (2, 0),
            (-2, 0.5),
            (2, 0.5),
        ]

        self._course_t_start: SE3 = SE3.Rt(SO3(), (2, 3, 1.5))
        self._course_t_gate: SE3 = SE3.Rt(SO3.Rz(0.0), (6.75, 3.4, 1.5))
        self._course_t_buoy: SE3 = SE3.Rt(SO3(), (12.7, 5.7, 2.5))
        self._buoy_t_buoy_scan: SE3 = SE3.Rt(SO3(), (-2, 0, 0))
        self._buoy_t_buoy_clear: SE3 = SE3.Rt(SO3(), (-2, 0, -1.5))
        self._course_t_octagon: SE3 = SE3.Rt(SO3(), (28.1, 23, 1.5))

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
            self._state = KFTransdec23State.GOTO_BUOY
            return goto.Goto(self._course_t_buoy, in_course=True)
        elif self._state == KFTransdec23State.GOTO_BUOY:
            self._state = KFTransdec23State.BUOY_SCAN
            return scan_translate.ScanTranslate(course_t_start=self._course_t_buoy * self._buoy_t_buoy_scan, points=self._buoy_scan_points)
        elif self._state == KFTransdec23State.BUOY_SCAN:
            # TODO: ADD CHECKING FOR MISSED BUOY DETECTION
            self._state = KFTransdec23State.BUOY_1
            return hit_buoy.HitBuoy(tag='buoy_abydos_1', timeout=30, frequency=10, distance=0.3, error_a=2, error_b=10, error_threshold=0.1, shoot_torpedo=None)
        elif self._state == KFTransdec23State.BUOY_1:
            self._state = KFTransdec23State.BUOY_2
            return hit_buoy.HitBuoy(tag='buoy_abydos_2', timeout=30, frequency=10, distance=0.3, error_a=2, error_b=10, error_threshold=0.1, shoot_torpedo=None)
        elif self._state == KFTransdec23State.BUOY_2:
            self._state = KFTransdec23State.CLEAR_BUOY
            return goto.Goto(self._course_t_buoy * self._buoy_t_buoy_clear, in_course=True)
        elif self._state == KFTransdec23State.CLEAR_BUOY:
            self._state = KFTransdec23State.GOTO_OCTAGON
            return goto.Goto(self._course_t_octagon, in_course=True)
        elif self._state == KFTransdec23State.GOTO_OCTAGON:
            self._state = KFTransdec23State.SURFACE
            return surface.Surface()
        else:
            return None