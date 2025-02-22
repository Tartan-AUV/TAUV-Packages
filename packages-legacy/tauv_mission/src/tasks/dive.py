import time
import rospy
from spatialmath import SE2, SE3, SO3
from dataclasses import dataclass
from tauv_util.spatialmath import flatten_se3
from tasks.task import Task, TaskResources, TaskStatus, TaskResult


class DiveStatus(TaskStatus):
    SUCCESS = 0
    FAILURE = 1


@dataclass
class DiveResult(TaskResult):
    status: DiveStatus


class Dive(Task):

    def __init__(self, delay: float, x_offset: float, y_offset: float):
        super().__init__()

        self._delay: float = delay
        self._x_offset = x_offset
        self._y_offset = y_offset

    def run(self, resources: TaskResources) -> DiveResult:
        odom_t_vehicle_initial = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
        odom_t_vehicle_initial = flatten_se3(odom_t_vehicle_initial)
        resources.transforms.set_a_to_b('kf/odom', 'kf/course', odom_t_vehicle_initial)

        # time.sleep(self._delay)
        start_time = time.time()
        while time.time() - start_time < self._delay:
            if self._check_cancel(resources): return DiveResult(DiveStatus.FAILURE)
            time.sleep(0.1)

        resources.motion.cancel()
        resources.motion.arm(True)

        odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')

        yaw_initial = odom_t_vehicle_initial.rpy()[2]
        R = SO3.Rz(yaw_initial)
        t = odom_t_vehicle.t.copy()
        t[2] = 1.0
        odom_t_vehicle_target = SE3.Rt(R, t)
        odom_t_vehicle_target = odom_t_vehicle_target * SE3.Tx(self._x_offset) * SE3.Ty(self._y_offset)

        resources.motion.goto(odom_t_vehicle_target)
        # resources.motion.goto_relative_with_depth(SE2(), self._depth)

        while True:
            if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                break

            if self._check_cancel(resources): return DiveResult(DiveStatus.FAILURE)

        return DiveResult(DiveStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()

