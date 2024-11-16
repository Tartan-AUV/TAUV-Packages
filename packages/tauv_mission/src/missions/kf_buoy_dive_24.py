from typing import Optional
from spatialmath import SE3, SO3, SE2
from math import pi
from missions.mission import Mission
from tasks.task import Task, TaskResult
from tasks import dive, gate_dead_reckon, goto, buoy_24, scan_translate
from enum import IntEnum

class State(IntEnum):
    START = 0
    DIVE = 1
    GATE_DEAD_RECKON = 2
    GOTO_WAYPOINT = 3
    CIRCLE_BUOY1 = 4
    SCAN = 5
    CIRCLE_BUOY2 = 6

class KFBuoyDive24(Mission):

    def __init__(self):
        self._state = State.START
        
        self._course_t_gate: SE3 = SE3.Rt(SO3(), (1.5, 0, 0))
        self._course_t_buoy_waypoint: SE3 = SE3.Rt(SO3(), (4.11, 3.43, 0.75))
        self._course_t_start: SE3 = SE3.Rt(SO3(), (4.11, 3.43, 0.75))
        self._scan_points: [(float, float)] = [(0.0, 0.0)]

    def entrypoint(self) -> Optional[Task]:
        self._state = State.DIVE
        return dive.Dive(20.0, 2, 0.0)
    
    def transition(self, task: Task, task_result: TaskResult) -> Optional[Task]:
        if self._state == State.DIVE:
            if task_result.status == dive.DiveStatus.SUCCESS:
                self._state = State.GATE_DEAD_RECKON
                return gate_dead_reckon.Gate(course_t_gate=self._course_t_gate)

        elif self._state == State.GATE_DEAD_RECKON:
            if task_result.status == gate_dead_reckon.GateStatus.SUCCESS:
                self._state = State.GOTO_WAYPOINT
                # return goto.Goto(self._course_t_buoy_waypoint)
                return None
        
        # elif self._state == State.GOTO_WAYPOINT:
        #     if task_result.status == goto.GotoStatus.SUCCESS:
        #         self._state = State.CIRCLE_BUOY1
        #         return buoy_24.CircleBuoy(tag='buoy_24')
        #
        # elif self._state == State.CIRCLE_BUOY1:
        #     if task_result.status == buoy_24.CircleBuoyStatus.SUCCESS:
        #         return None
        #     else:
        #         self._state = State.SCAN
        #         return scan_translate.ScanTranslate(self._course_t_start, self._scan_points)
        #
        # elif self._state == State.SCAN:
        #     if task_result.status == scan_translate.ScanTranslateStatus.SUCCESS:
        #         self._state = State.CIRCLE_BUOY2
        #         return buoy_24.CircleBuoy('buoy_24')
        #
        else:
            return None