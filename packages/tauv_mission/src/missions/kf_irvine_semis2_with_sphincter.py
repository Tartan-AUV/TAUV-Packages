from typing import Optional
from spatialmath import SE3, SO3, SE2
from math import pi
from missions.mission import Mission
from tasks.task import Task, TaskResult
from tasks import dive, gate_dead_reckon, goto, goto_relative_with_depth, buoy_24, surface, buoy_24_dead_reckon
from enum import IntEnum

class State(IntEnum):
    DIVE=10
    START = 0
    GATE_DEAD_RECKON = 1
    GOTO_BUOY = 2
    SURVEY_BUOY = 9
    BUOY = 3
    GOTO_OCTAGON = 4
    OCTAGON_SURFACE = 5
    OCTAGON_DIVE = 6
    OCTAGON_FACE_BUOY = 7
    FINAL_SURFACE = 8


class KFIrvineSemis2(Mission):

    def __init__(self):
        self._state = State.START

        # Measurements in NED frame
        self._gate_t_buoy: SE3 = SE3.Rt(SO3(), (4.80, -4.11, 0.75))
        self._gate_t_octagon: SE3 = SE3.Rt(SO3(), (13.72, -2.74, 0))

        self._wall_t_gate: SE3 = SE3.Rt(SO3(), (5.07, 0, 0))

        self._wall_t_vehicle_rear: SE3 = SE3.Rt(SO3(), (0.3048, 0, 0)) 
        self._vehicle_rear_t_course: SE3 = SE3.Rt(SO3(), (0.40, 0, 0))

        self._wall_t_course: SE3 = self._wall_t_vehicle_rear * self._vehicle_rear_t_course

        # Transforms to course frame
        self._course_t_gate = self._wall_t_course.inv() * self._wall_t_gate
        self._course_t_buoy = self._course_t_gate * self._gate_t_buoy
        self._course_t_octagon = self._course_t_gate * self._gate_t_octagon

        self._course_t_buoy_approach = self._course_t_buoy * SE3.Rt(SO3.Rz(-1.57), (0, 3, 0))
        self._course_t_octagon_approach = self._course_t_octagon * SE3.Rt(SO3(), (0, 0, 0.75))
        self._course_t_octagon_approach_face_buoy = self._course_t_octagon_approach * SE3.Rt(SO3(), (-6.86,0,0))

    def entrypoint(self) -> Optional[Task]:
        self._state = State.DIVE
        return dive.Dive(20.0, 2.0, 0.0)

    def transition(self, task: Task, task_result: TaskResult) -> Optional[Task]:
        if self._state == State.DIVE:
            if task_result.status == dive.DiveStatus.SUCCESS:
                self._state = State.GATE_DEAD_RECKON
                return gate_dead_reckon.Gate(course_t_gate=self._course_t_gate,
                                             travel_offset_y=-0.75)

        elif self._state == State.GATE_DEAD_RECKON:
            if task_result.status == gate_dead_reckon.GateStatus.SUCCESS:
                self._state = State.GOTO_OCTAGON
                return goto.Goto(self._course_t_octagon_approach, in_course=True)

        elif self._state == State.GOTO_OCTAGON:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.OCTAGON_SURFACE
                return goto_relative_with_depth.GotoRelativeWithDepth(SE2(), 0)

        elif self._state == State.OCTAGON_SURFACE:
            if task_result.status == goto_relative_with_depth.GotoRelativeWithDepthStatus.SUCCESS:
                self._state = State.OCTAGON_DIVE
                return goto.Goto(self._course_t_octagon_approach, in_course=True)

        elif self._state == State.OCTAGON_DIVE:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.OCTAGON_FACE_BUOY
                return goto.Goto(self._course_t_octagon_approach_face_buoy, in_course=True)

        elif self._state == State.OCTAGON_FACE_BUOY:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.GOTO_BUOY
                return goto.Goto(self._course_t_buoy_approach, in_course=True)
        
        elif self._state == State.GOTO_BUOY:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.BUOY
                # return buoy_24.CircleBuoy('buoy_24', circle_radius=3.0,
                #                                         circle_ccw=False,
                #                                         waypoint_every_n_meters=0.75,
                #                                         latch_buoy=False)
                return buoy_24_dead_reckon.CircleBuoyDeadReckon(self._course_t_buoy,
                                                                circle_depth=0.7)


        elif self._state == State.BUOY:
            if task_result.status == buoy_24.CircleBuoyStatus.SUCCESS:
                self._state = State.FINAL_SURFACE
                return surface.Surface()

        else:
            return None

