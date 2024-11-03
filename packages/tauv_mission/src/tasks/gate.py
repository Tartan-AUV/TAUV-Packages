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
    TIMEOUT = 2
    GATE_NOT_FOUND = 3


@dataclass
class GateResult(TaskResult):
    status: GateStatus


class Gate(Task):

    def __init__(self, timeout: float):
        super().__init__()

        self._timeout = timeout

    def run(self, resources: TaskResources) -> GateResult:
        timeout_time = rospy.Time.now() + rospy.Duration.from_sec(self._timeout)

        odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')

        scan_thetas = np.linspace(-pi / 2, pi / 2, 4)

        for scan_theta in scan_thetas:
            print("SCAN THETA", scan_thetas
                  )
            odom_t_vehicle_goal = odom_t_vehicle * SE3.Rz(scan_theta)

            resources.motion.goto(odom_t_vehicle_goal)

            if self._spin_cancel(resources, lambda: resources.motion.wait_until_complete(
                    timeout=rospy.Duration.from_sec(0.1)), timeout_time):
                return GateResult(status=GateStatus.TIMEOUT)

            time.sleep(2.0)

        resources.motion.goto(odom_t_vehicle)

        if self._spin_cancel(resources, lambda: resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)), timeout_time):
            return GateResult(status=GateStatus.TIMEOUT)

        gate_detection = resources.map.find_closest('gate', odom_t_vehicle.t)

        if gate_detection is None:
            return GateResult(status=GateStatus.GATE_NOT_FOUND)

        odom_t_gate = gate_detection.pose

        gate_t_spin = SE3.Rt(SO3(), (-2, 0, 0.5))
        gate_t_through = SE3.Rt(SO3(), (2, 0, 0.5))

        odom_t_spin = odom_t_gate * gate_t_spin
        odom_t_through = odom_t_gate * gate_t_through

        resources.transforms.set_a_to_b('kf/odom', 'gate', odom_t_gate)
        resources.transforms.set_a_to_b('kf/odom', 'spin', odom_t_spin)
        resources.transforms.set_a_to_b('kf/odom', 'through', odom_t_through)

        resources.motion.goto(odom_t_spin)

        if self._spin_cancel(resources, lambda: resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)), timeout_time):
            return GateResult(status=GateStatus.TIMEOUT)

        spin_thetas = np.concatenate((
            np.linspace(0, 2 * pi, 4),
            np.linspace(0, 2 * pi, 4),
        ))

        for spin_theta in spin_thetas:
            odom_t_vehicle_goal = odom_t_spin * SE3.Rz(spin_theta, unit='rad')

            resources.motion.goto(odom_t_vehicle_goal)

            if self._spin_cancel(resources, lambda: resources.motion.wait_until_complete(
                    timeout=rospy.Duration.from_sec(0.1)), timeout_time):
                return GateResult(status=GateStatus.TIMEOUT)

        resources.motion.goto(odom_t_through)

        if self._spin_cancel(resources, lambda: resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)), timeout_time):
            return GateResult(status=GateStatus.TIMEOUT)

        return GateResult(status=GateStatus.SUCCESS)



    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()
