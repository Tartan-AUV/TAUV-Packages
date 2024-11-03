import rospy
import numpy as np
from math import pi
from spatialmath import SE3, SO3
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
import time
from tauv_util.spatialmath import flatten_se3


class GateStatus(TaskStatus):
    SUCCESS = 0
    CANCELLED = 1
    GATE_NOT_FOUND = 2


@dataclass
class GateResult(TaskResult):
    status: GateStatus


class Gate(Task):

    def __init__(self, course_t_gate: SE3, travel_offset_y: float):
        super().__init__()

        self._course_t_gate = course_t_gate
        self._travel_offset_y = travel_offset_y

    def run(self, resources: TaskResources) -> GateResult:
        odom_t_course = resources.transforms.get_a_to_b('kf/odom', 'kf/course')
        odom_t_gate = odom_t_course * self._course_t_gate

        gate_t_spin = SE3.Rt(SO3(), (-2, 0, 1.0))
        gate_t_through = SE3.Rt(SO3(), (0, self._travel_offset_y, 1.0))

        odom_t_spin = odom_t_gate * gate_t_spin
        odom_t_through = odom_t_gate * gate_t_through

        resources.transforms.set_a_to_b('kf/odom', 'gate', odom_t_gate)
        resources.transforms.set_a_to_b('kf/odom', 'spin', odom_t_spin)
        resources.transforms.set_a_to_b('kf/odom', 'through', odom_t_through)

        resources.motion.goto(odom_t_spin, params=resources.motion.get_trajectory_params("rapid"))

        while True:
            if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                break

            if self._check_cancel(resources): return GateResult(status=GateStatus.CANCELLED)

        spin_thetas = np.array([0, pi, 0, pi, 0])

        for spin_theta in spin_thetas:
            odom_t_vehicle_goal = odom_t_spin * SE3.Rz(spin_theta, unit='rad')

            resources.motion.goto(odom_t_vehicle_goal, params=resources.motion.get_trajectory_params("rapid"))

            while True:
                if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                    break

                if self._check_cancel(resources): return GateResult(status=GateStatus.CANCELLED)

        resources.motion.goto(odom_t_through, resources.motion.get_trajectory_params("rapid"))

        while True:
            if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                break

            if self._check_cancel(resources): return GateResult(status=GateStatus.CANCELLED)

        return GateResult(status=GateStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()
