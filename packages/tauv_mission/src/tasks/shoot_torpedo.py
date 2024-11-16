import rospy
import numpy as np
from spatialmath import SE3, SO3
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
from tauv_util.spatialmath import flatten_se3


class ShootTorpedoStatus(TaskStatus):
    SUCCESS = 0
    CANCELLED = 1
    TIMEOUT = 2
    TARGET_NOT_FOUND = 3
    TARGET_LOST = 4

@dataclass
class ShootTorpedoResult(TaskResult):
    status: ShootTorpedoStatus


class ShootTorpedo(Task):

    def __init__(self, tag: str, torpedo: int, timeout: float, frequency: float, distance: float, error_factor: float, error_threshold: float):
        super().__init__()

        self._tag = tag
        self._timeout = timeout
        self._torpedo = torpedo
        self._period = rospy.Duration.from_sec(1 / frequency)
        self._distance = distance
        self._error_factor = error_factor
        self._error_threshold = error_threshold

    def run(self, resources: TaskResources) -> ShootTorpedoResult:
        timeout_time = rospy.Time.now() + rospy.Duration.from_sec(self._timeout)

        odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
        target_detection = resources.map.find_closest(self._tag, odom_t_vehicle.t)

        target_t_torpedo = SE3.Tx(-self._distance)
        vehicle_t_torpedo = resources.transforms.get_a_to_b('kf/vehicle', 'kf/torpedo')

        if target_detection is None:
            return ShootTorpedoResult(status=ShootTorpedoStatus.TARGET_NOT_FOUND)

        resources.motion.cancel()

        odom_t_target = target_detection.pose

        while rospy.Time.now() < timeout_time:
            odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
            target_detection = resources.map.find_closest(self._tag, odom_t_target.t)

            if target_detection is None:
                return ShootTorpedoResult(status=ShootTorpedoStatus.TARGET_LOST)

            odom_t_target = target_detection.pose
            odom_t_vehicle_goal = odom_t_target * target_t_torpedo * vehicle_t_torpedo.inv()
            odom_t_vehicle_goal = flatten_se3(odom_t_vehicle_goal)

            error = np.linalg.norm(odom_t_vehicle.t - odom_t_vehicle_goal.t) + abs(odom_t_vehicle.rpy()[2] - odom_t_vehicle_goal.rpy()[2])
            if error < self._error_threshold:
                break

            resources.transforms.set_a_to_b('kf/odom', 'torpedo_target', odom_t_target)
            resources.transforms.set_a_to_b('kf/odom', 'vehicle_goal', odom_t_vehicle_goal)

            target_t_vehicle_goal = odom_t_target.inv() * odom_t_vehicle_goal

            orthogonal_error = np.linalg.norm(odom_t_vehicle_goal.t[1:3] - odom_t_vehicle.t[1:3])\
                + abs(odom_t_vehicle_goal.rpy()[2] - odom_t_vehicle.rpy()[2])

            x = -(1 - np.exp(-self._error_factor * orthogonal_error)) * target_t_vehicle_goal.t[0]

            target_t_vehicle_target = SE3.Rt(
                target_t_vehicle_goal.R,
                (x, target_t_vehicle_goal.t[1], target_t_vehicle_goal.t[2])
            )

            odom_t_vehicle_target = odom_t_target * target_t_vehicle_target

            resources.transforms.set_a_to_b('kf/odom', 'vehicle_target', odom_t_vehicle_target)

            resources.motion.goto(odom_t_vehicle_target, params=resources.motion.get_trajectory_params("feedback"))

            if self._check_cancel(resources): return ShootTorpedoResult(ShootTorpedoStatus.CANCELLED)

            rospy.sleep(self._period)

        resources.actuators.shoot_torpedo(self._torpedo)

        resources.motion.goto_relative(SE3.Rt(SO3(), (-1, 0, 0)))

        if not self._spin_cancel(resources, lambda: resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1))):
            return ShootTorpedoResult(status=ShootTorpedoStatus.TIMEOUT)

        return ShootTorpedoResult(status=ShootTorpedoStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()