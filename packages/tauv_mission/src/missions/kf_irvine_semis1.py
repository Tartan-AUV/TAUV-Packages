from typing import Optional
from spatialmath import SE3, SO3, SE2
from math import pi
from missions.mission import Mission
from tasks.task import Task, TaskResult
from tasks import dive, goto, gate, scan_rotate, yaw_roll, buoy_24, surface
from enum import IntEnum

class State(IntEnum):
    START = 0
    DIVE = 1
    ROLL = 2
    GOTO_GATE = 3
    GATE = 4
    GOTO_BUOY = 5
    BUOY = 6
    GOTO_OCTAGON = 7
    OCTAGON_SURFACE = 8
    OCTAGON_DIVE = 9
    OCTAGON_FACE_BUOY = 11
    FINAL_SURFACE = 10


class KFIrvineSemis1(Mission):

    def __init__(self):
        self._state = State.START

        ### START
        self._gate_t_buoy: SE3 = SE3.Rt(SO3(), (1, 3.43, 0.75))
        self._gate_t_octagon: SE3 = SE3.Rt(SO3(), (11.89, 1.37, 0))

        self._wall_t_gate: SE3 = SE3.Rt(SO3(), (5.94, 0, 0))

        self._wall_t_vehicle_rear: SE3 = SE3.Rt(SO3(), (0.3048, 0, 0)) # how far rylan puts it out judged from space bw wall and back
        self._vehicle_rear_t_odom: SE3 = SE3.Rt(SO3(), (0.40, 0, 0))

        self._wall_t_odom: SE3 = self._wall_t_vehicle_rear * self._vehicle_rear_t_odom

        # Transform everything into odom frame
        self._odom_t_gate = self._wall_t_odom.inv() * self._wall_t_gate
        self._odom_t_buoy = self._odom_t_gate * self._gate_t_buoy
        self._odom_t_octagon = self._odom_t_gate * self._gate_t_octagon

        self._odom_t_gate_approach_front = self._odom_t_gate * SE3.Rt(SO3(), (-1, -0.75, 1.3))
        self._odom_t_gate_approach_back = self._odom_t_gate * SE3.Rt(SO3(), (1, -0.75, 1.3))
        self._odom_t_buoy_approach = self._odom_t_buoy * SE3.Rt(SO3.Rz(1.57), (0, -3, 0))
        self._odom_t_octagon_approach = self._odom_t_octagon * SE3.Rt(SO3(), (0, 0, 0.75))
        self._odom_t_octagon_approach_face_buoy = self._odom_t_octagon_approach * SE3.Rt(SO3.Rz(3.14), (0,0,0))
        ###

        # self._gate_pose: SE3 = SE3.Rt(SO3(), (4, -0.5, 0))# center is top of gate
        # # self._buoy_pose: SE3 = SE3.Rt(SO3(), (7, 3, 1))
        # self._octagon_pose: SE3 = SE3.Rt(SO3(), (19, 0.75, 0))
        #
        # self._buoy_waypoint: SE3 = SE3.Rt(SO3.Rz(1.57), (7.5, 0, 1.5))
        #
        # self._goto_gate_offset: SE3 = SE3.Rt(SO3(), (-1, 0.75, 1))
        # self._after_gate_offset: SE3 = SE3.Rt(SO3(), (1, 0.75, 1))
        #
        # self._octagon_before_offset: SE3 = SE3.Rt(SO3(), (0, 0, 1))

    def entrypoint(self) -> Optional[Task]:
        self._state = State.DIVE
        return dive.Dive(20.0, 2, 0.0)

    def transition(self, task: Task, task_result: TaskResult) -> Optional[Task]:
        if self._state == State.DIVE:
            if task_result.status == dive.DiveStatus.SUCCESS:
                # self._state = State.ROLL
                # return yaw_roll.YawRoll()
                self._state = State.GOTO_GATE
                return goto.Goto(self._odom_t_gate_approach_front)

        # elif self._state == State.ROLL:
        #     if task_result.status == yaw_roll.YawRollStatus.SUCCESS:
        #         self._state = State.GOTO_GATE
        #         return goto.Goto(self._goto_gate_offset * self._gate_pose)

        elif self._state == State.GOTO_GATE:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.GATE
                return goto.Goto(self._odom_t_gate_approach_back)

        elif self._state == State.GATE:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.GOTO_OCTAGON
                return goto.Goto(self._odom_t_octagon_approach)

        elif self._state == State.GOTO_OCTAGON:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.OCTAGON_SURFACE
                return surface.Surface()

        elif self._state == State.OCTAGON_SURFACE:
            if task_result.status == surface.SurfaceStatus.SUCCESS:
                self._state = State.OCTAGON_DIVE
                return goto.Goto(self._odom_t_octagon_approach)

        elif self._state == State.OCTAGON_DIVE:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.OCTAGON_FACE_BUOY
                return goto.Goto(self._odom_t_octagon_approach_face_buoy)

        elif self._state == State.OCTAGON_FACE_BUOY:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.BUOY
                return goto.Goto(self._odom_t_buoy_approach)

        elif self._state == State.GOTO_BUOY:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.BUOY
                return buoy_24.CircleBuoy('buoy_24', circle_radius=2.5, waypoint_every_n_meters=0.75, circle_ccw=False)

        elif self._state == State.BUOY:
            if task_result.status == buoy_24.CircleBuoyStatus.SUCCESS:
                self._state = State.FINAL_SURFACE
                return surface.Surface()

        else:
            return None

