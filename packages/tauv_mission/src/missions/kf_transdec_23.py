from typing import Optional
from spatialmath import SE3, SO3, SE2
from missions.mission import Mission
from tasks.task import Task, TaskResult
from tasks import dive, goto, gate, detect_pinger, hit_buoy, gate_dead_reckon
from enum import IntEnum


class KFTransdec23State(IntEnum):
    UNKNOWN = 0
    DIVE = 1
    GOTO_GATE = 2
    GATE = 3
    GOTO_BUOY = 4
    BUOY_1 = 5
    BUOY_2 = 6
    DETECT_PINGER = 7

class KFTransdec23(Mission):

    def __init__(self):
        self._state = KFTransdec23State.UNKNOWN

        self._start_pose: SE3 = SE3.Rt(SO3(), (3, 0, 0))
        self._buoy_pose: SE3 = SE3.Rt(SO3(), (6, 6, 0))
        self._pinger_frequency: int = 30000

    def entrypoint(self) -> Optional[Task]:
        self._state = KFTransdec23State.DIVE
        return dive.Dive(delay=20.0)

    def transition(self, task: Task, task_result: TaskResult) -> Optional[Task]:
        if self._state == KFTransdec23State.DIVE \
            and task_result.status == dive.DiveStatus.SUCCESS:
            self._state = KFTransdec23State.GOTO_GATE
            return goto.Goto(self._start_pose, in_course=True)
        elif self._state == KFTransdec23State.GOTO_GATE \
            and task_result.status == goto.GotoStatus.SUCCESS:
            self._state = KFTransdec23State.GATE
            return gate.Gate(timeout=300)
        elif self._state == KFTransdec23State.GATE:
            if task_result.status == gate.GateStatus.GATE_NOT_FOUND:
                return gate_dead_reckon.GateDeadReckon()
            elif task_result.status == gate.GateStatus.SUCCESS:
                self._state = KFTransdec23State.GOTO_BUOY
                return goto.Goto(self._buoy_pose, in_course=True)
        elif self._state == KFTransdec23State.GOTO_BUOY:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = KFTransdec23State.BUOY_1
                return hit_buoy.HitBuoy('buoy-earth-1', timeout=30, distance=0.2, error_threshold=0.1)
        elif self._state == KFTransdec23State.BUOY_1:
            if task_result.status == hit_buoy.HitBuoy.SUCCESS:
                self._state = KFTransdec23State.BUOY_2
                return hit_buoy.HitBuoy('buoy-earth-2', timeout=30, distance=0.2, error_threshold=0.1)
        elif self._state == KFTransdec23State.BUOY_2:
            if task_result.status == hit_buoy.HitBuoy.SUCCESS:
                self._state = KFTransdec23State.DETECT_PINGER
                return detect_pinger.DetectPinger(self._pinger_frequency, timeout=60, depth=2.0)
        else:
            return None