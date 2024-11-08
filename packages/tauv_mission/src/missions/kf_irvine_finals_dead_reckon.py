from typing import Optional
from spatialmath import SE3, SO3, SE2
from math import pi
import yaml
import platform
from actuator_client import ActuatorClient
from missions.mission import Mission
from missions.coordinates import Course
from tasks.task import Task, TaskResult
from tasks import dive, gate_dead_reckon, goto, goto_relative_with_depth, surface, buoy_24_dead_reckon, barrel_roll
from enum import IntEnum
# variables
COURSE = "finals"
GATE_SIDE = -1 # -1 = left, 1 = right
# HAS TO BE CCW
if platform.machine() == "aarch64":
    tauv_ros_packages_dir = "/shared/tauv_ws/src/TAUV-ROS-Packages/"
else:
    tauv_ros_packages_dir = "/home/gleb/catkin_ws/src/TAUV-ROS-Packages/"
yaml_dir = tauv_ros_packages_dir + "src/packages/tauv_mission/src/missions/irvine_24.yaml"
with open(yaml_dir, "r") as file:
    data = yaml.safe_load(file)
coords = Course(**data[COURSE])
class State(IntEnum):
    # start run
    START = 0
    DIVE = 1
    GATE_DEAD_RECKON = 2
    # buoy
    BUOY_GOTO = 3
    BUOY_DEAD_RECKON = 4
    BUOY_CIRCLE = 5
    BUOY_BACKOFF = 19
    # torpedo:
    TORPEDO_GOTO = 6
    TORPEDO_DEAD_RECKON = 7
    TORPEDO_BACKOFF = 20
    # octagon
    OCTAGON_GOTO = 8
    OCTAGON_SURFACE = 9
    OCTAGON_DIVE = 10
    SAMPLE_PICKUP_DEAD_RECKON = 11
    SAMPLE_DROP = 12
    OCTAGON_BACKOFF = 21
    # marker
    MARKER_GOTO = 13
    MARKER_DEAD_RECKON = 14
    MARKER_BACKOFF = 22
    # end run
    STYLE_GOTO = 15
    STYLE = 16
    FINAL_SURFACE = 17
class KFIrvineFinals(Mission):
    def __init__(self):
        # initializing
        self._state = State.START
        self._actuators = ActuatorClient()
        # transformation of coordinates to course frame
        self._wall_t_rear: SE3 = SE3.Rt(SO3(), (coords._wall_t_rear.x, coords._wall_t_rear.y, coords._wall_t_rear.z))
        self._rear_t_course: SE3 = SE3.Rt(SO3(), (coords._rear_t_course.x, coords._rear_t_course.y, coords._rear_t_course.z))
        self._wall_t_course: SE3 = self._wall_t_rear * self._rear_t_course
        self._wall_t_ref: SE3 = SE3.Rt(SO3(), (coords._wall_t_ref.x, coords._wall_t_ref.y, coords._wall_t_ref.z))
        self._ref_t_gate: SE3 = SE3.Rt(SO3(), (coords._ref_t_gate.x, coords._ref_t_gate.y, coords._ref_t_gate.z))
        self._wall_t_gate: SE3 = self._wall_t_ref * SE3.Rt(SO3.Rz(coords._ref_t_gate.deg, unit="deg"), [0, 0, 0])
        self._course_t_ref = self._wall_t_course.inv() * self._wall_t_ref
        self._course_t_gate = self._wall_t_course.inv() * self._wall_t_gate
        self._gate_offset_y = GATE_SIDE*0.75
        # buoy
        self._ref_t_buoy: SE3 = SE3.Rt(SO3(), (coords._ref_t_buoy.x, coords._ref_t_buoy.y, coords._ref_t_buoy.z))
        self._course_t_buoy = self._course_t_ref * self._ref_t_buoy
        self._course_t_buoy_approach = self._course_t_buoy * SE3.Rt(SO3.Rz(coords._buoy_t_approach.deg), (coords._buoy_t_approach.x, coords._buoy_t_approach.y, coords._buoy_t_approach.z))
        self._course_t_buoy_backoff = self._course_t_buoy_approach * SE3.Rt(SO3.Rz(coords._buoy_approach_t_backoff.deg), (coords._buoy_approach_t_backoff.x, coords._buoy_approach_t_backoff.y, coords._buoy_approach_t_backoff.z))
        # torpedo
        self._ref_t_torpedo: SE3 = SE3.Rt(SO3.Rz(coords._ref_t_torpedo.deg, unit="deg"), (coords._ref_t_torpedo.x, coords._ref_t_torpedo.y, coords._ref_t_torpedo.z))
        self._course_t_torpedo = self._course_t_ref * self._ref_t_torpedo
        self._course_t_torpedo_approach = self._course_t_torpedo * SE3.Rt(SO3.Rz(coords._torpedo_t_approach.deg), (coords._torpedo_t_approach.x, coords._torpedo_t_approach.y, coords._torpedo_t_approach.z))
        self._course_t_torpedo_backoff = self._course_t_torpedo_approach * SE3.Rt(SO3.Rz(coords._torpedo_approach_t_backoff.deg), (coords._torpedo_approach_t_backoff.x, coords._torpedo_approach_t_backoff.y, coords._torpedo_approach_t_backoff.z))
        # octagon
        self._ref_t_octagon: SE3 = SE3.Rt(SO3(), (coords._ref_t_octagon.x, coords._ref_t_octagon.y, coords._ref_t_octagon.z))
        self._course_t_octagon = self._course_t_ref * self._ref_t_octagon
        self._course_t_octagon_approach = self._course_t_octagon * SE3.Rt(SO3.Rz(coords._octagon_t_approach.deg), (coords._octagon_t_approach.x, coords._octagon_t_approach.y, coords._octagon_t_approach.z))
        self._course_t_octagon_backoff = self._course_t_octagon_approach * SE3.Rt(SO3.Rz(coords._octagon_approach_t_backoff.deg), (coords._octagon_approach_t_backoff.x, coords._octagon_approach_t_backoff.y, coords._octagon_approach_t_backoff.z))
        # marker
        self._ref_t_marker: SE3 = SE3.Rt(SO3(), (coords._ref_t_marker.x, coords._ref_t_marker.y, coords._ref_t_marker.z))
        self._course_t_marker = self._course_t_ref * self._ref_t_marker
        self._course_t_marker_approach = self._course_t_marker * SE3.Rt(SO3.Rz(coords._marker_t_approach.deg), (coords._marker_t_approach.x, coords._marker_t_approach.y, coords._marker_t_approach.z))
        self._course_t_marker_backoff = self._course_t_marker_approach * SE3.Rt(SO3.Rz(coords._marker_approach_t_backoff.deg), (coords._marker_approach_t_backoff.x, coords._marker_approach_t_backoff.y, coords._marker_approach_t_backoff.z))
        # style
        self._course_t_style = self._course_t_gate * SE3.Rt(SO3(), (coords._gate_t_style.x, coords._gate_t_style.y, coords._gate_t_style.z))
    def entrypoint(self) -> Optional[Task]:
        self._state = State.DIVE
        dive_delay = 20.0
        return dive.Dive(dive_delay, 2.0, 0.0)
    def transition(self, task: Task, task_result: TaskResult) -> Optional[Task]:
        print(self._state)
        ########### START RUN ############
        if self._state == State.DIVE:
            if task_result.status == dive.DiveStatus.SUCCESS:
                self._state = State.GATE_DEAD_RECKON
                return gate_dead_reckon.Gate(course_t_gate=self._course_t_gate, travel_offset_y=self._gate_offset_y)
        elif self._state == State.GATE_DEAD_RECKON:
            if task_result.status == gate_dead_reckon.GateStatus.SUCCESS:
                self._state = State.TORPEDO_GOTO
                return goto.Goto(self._course_t_torpedo_approach, in_course=True)
        ########### TORPEDO ############
        elif self._state == State.TORPEDO_GOTO:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.TORPEDO_DEAD_RECKON
                return goto.Goto(self._course_t_torpedo_approach, in_course=True)
        elif self._state == State.TORPEDO_DEAD_RECKON:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._actuators.shoot_torpedo(1) # ADD TASKS
                self._actuators.shoot_torpedo(0)
                self._state = State.TORPEDO_BACKOFF
                return goto.Goto(self._course_t_torpedo_backoff, in_course=True)
        elif self._state == State.TORPEDO_BACKOFF:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.OCTAGON_GOTO
                return goto.Goto(self._course_t_octagon_approach, in_course=True)
        ########### OCTAGON ############
        elif self._state == State.OCTAGON_GOTO:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.OCTAGON_SURFACE
                return goto.Goto(self._course_t_octagon_approach, in_course=True)
        # elif self._state == State.SAMPLE_PICKUP_DEAD_RECKON:
        #     if task_result.status == goto.GotoStatus.SUCCESS:
        #         self._actuators.close_sphincter()
        #         self._state = State.OCTAGON_SURFACE
        #         return goto_relative_with_depth.GotoRelativeWithDepth(SE2(), 0)
        elif self._state == State.OCTAGON_SURFACE:
            if task_result.status == goto_relative_with_depth.GotoRelativeWithDepthStatus.SUCCESS:
                self._state = State.OCTAGON_DIVE
                return goto.Goto(self._course_t_octagon_approach, in_course=True)
        elif self._state == State.OCTAGON_DIVE:
            if task_result.status == goto.GotoStatus.SUCCESS:
                #self._actuators.open_sphincter()
                self._state = State.OCTAGON_BACKOFF
                return goto.Goto(self._course_t_octagon_backoff, in_course=True)
        elif self._state == State.OCTAGON_BACKOFF:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.MARKER_GOTO
                return goto.Goto(self._course_t_marker_approach, in_course=True)
# can insert some way to pick up second piece if the first was retrieved with vision and if it is seen
        ########### MARKER ############
        elif self._state == State.MARKER_GOTO:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.MARKER_DEAD_RECKON
                return goto.Goto(self._course_t_marker_approach, in_course=True)
        elif self._state == State.MARKER_DEAD_RECKON:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._actuators.drop_marker(0) # ADD TASK
                self._actuators.drop_marker(1)
                self._state = State.MARKER_BACKOFF
                return goto.Goto(self._course_t_marker_backoff, in_course=True)
        elif self._state == State.MARKER_BACKOFF:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.BUOY_GOTO
                return goto.Goto(self._course_t_buoy_approach, in_course=True)
        ########### BUOY ############
        elif self._state == State.BUOY_GOTO:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.BUOY_DEAD_RECKON
                return goto.Goto(self._course_t_buoy_approach, in_course=True)
        elif self._state == State.BUOY_DEAD_RECKON:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.BUOY_CIRCLE
                return buoy_24_dead_reckon.CircleBuoyDeadReckon(self._course_t_buoy,
                 circle_radius=2.5, circle_ccw=True, waypoint_every_n_meters=1.0,
                 circle_depth=0.7, n_torpedos=0)
        elif self._state == State.BUOY_CIRCLE:
            if task_result.status == buoy_24_dead_reckon.CircleBuoyDeadReckonStatus.SUCCESS:
                self._state = State.BUOY_BACKOFF
                return goto.Goto(self._course_t_buoy_backoff, in_course=True)
        elif self._state == State.BUOY_BACKOFF:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.STYLE_GOTO
                return goto.Goto(self._course_t_style, in_course=True)
        ########### STYLE ############
        elif self._state == State.STYLE_GOTO:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.STYLE
                return barrel_roll.BarrelRoll()
        elif self._state == State.STYLE:
            if task_result.status == barrel_roll.BarrelRollStatus.SUCCESS:
                self._state = State.FINAL_SURFACE
                return surface.Surface()
        else:
            return None
