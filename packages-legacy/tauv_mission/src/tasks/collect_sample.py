import rospy
import numpy as np
from spatialmath import SE3, SO3, SE2
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
from tauv_util.spatialmath import flatten_se3
from typing import Optional


class CollectSampleStatus(TaskStatus):
    SUCCESS = 0
    CANCELLED = 1
    TIMEOUT = 2
    SAMPLE_NOT_FOUND = 3
    SAMPLE_LOST = 4


@dataclass
class CollectSampleResult(TaskResult):
    status: CollectSampleStatus


class CollectSample(Task):

    def __init__(self, tag: str, timeout: float, frequency: float, distance: float, error_a: float, error_b: float, error_threshold: float):
        super().__init__()

        self._tag = tag
        self._timeout = timeout
        self._period = rospy.Duration.from_sec(1 / frequency)
        self._distance = distance
        self._error_a = error_a
        self._error_b = error_b
        self._error_threshold = error_threshold

    def run(self, resources: TaskResources) -> CollectSampleResult:
        timeout_time = rospy.Time.now() + rospy.Duration.from_sec(self._timeout)

        odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
        sample_detection = resources.map.find_closest(self._tag, odom_t_vehicle.t)

        vehicle_t_sphincter = resources.transforms.get_a_to_b('kf/vehicle', 'kf/sphincter')

        sample_t_vehicle_goal = SE3.Tz(-self._distance) * vehicle_t_sphincter.inv()

        if sample_detection is None:
            print("NOT FOUND")
            return CollectSampleResult(status=CollectSampleStatus.SAMPLE_NOT_FOUND)

        resources.motion.cancel()

        odom_t_sample = sample_detection.pose

        if np.linalg.norm(odom_t_sample.t - odom_t_vehicle.t) > 5:
            return CollectSampleResult(status=CollectSampleStatus.SAMPLE_NOT_FOUND)

        while rospy.Time.now() < timeout_time:
            odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')

            sample_detection = resources.map.find_closest(self._tag, odom_t_sample.t)

            if sample_detection is None:
                return CollectSampleResult(status=CollectSampleStatus.SAMPLE_LOST)

            odom_t_sample = sample_detection.pose
            odom_t_sample = SE3.Rt(odom_t_vehicle.R, odom_t_sample.t)
            odom_t_vehicle_goal = flatten_se3(odom_t_sample * sample_t_vehicle_goal)

            if np.linalg.norm(odom_t_vehicle.t - odom_t_vehicle_goal.t) < self._error_threshold:
                resources.actuators.close_sphincter()
                break

            sample_t_vehicle = odom_t_sample.inv() * odom_t_vehicle

            orthogonal_error = np.linalg.norm(sample_t_vehicle.t[0:2] - sample_t_vehicle_goal.t[0:2])

            z = -self._error_a * (1 - np.exp(-self._error_b * orthogonal_error))

            odom_t_vehicle_target = SE3.Rt(odom_t_vehicle_goal.R, np.array([
                odom_t_vehicle_goal.t[0],
                odom_t_vehicle_goal.t[1],
                z + odom_t_vehicle_goal.t[2],
            ]))

            resources.transforms.set_a_to_b('kf/odom', 'sample', odom_t_sample)

            resources.transforms.set_a_to_b('kf/odom', 'vehicle_goal', odom_t_vehicle_goal)

            resources.transforms.set_a_to_b('kf/odom', 'vehicle_target', odom_t_vehicle_target)

            resources.motion.goto(odom_t_vehicle_target, params=resources.motion.get_trajectory_params("feedback"))

            if self._check_cancel(resources): return CollectSampleResult(status=CollectSampleStatus.CANCELLED)

            rospy.sleep(self._period)

        resources.motion.goto_relative_with_depth(SE2(), 0)

        if self._spin_cancel(resources, lambda: resources.motion.wait_until_complete(
                timeout=rospy.Duration.from_sec(0.1)), timeout_time):
            return CollectSampleResult(status=CollectSampleStatus.TIMEOUT)

        resources.motion.arm(False)

        rospy.sleep(5.0)

        resources.motion.arm(True)

        resources.motion.cancel()
        resources.motion.goto_relative_with_depth(SE2(), 1.5)

        if self._spin_cancel(resources, lambda: resources.motion.wait_until_complete(
                timeout=rospy.Duration.from_sec(0.1)), timeout_time):
            return CollectSampleResult(status=CollectSampleStatus.TIMEOUT)

        rospy.sleep(5.0)

        # for _ in range(5):
        #     resources.actuators.open_sphincter()
        #     resources.actuators.close_sphincter()

        resources.actuators.open_sphincter()

        resources.motion.goto_relative_with_depth(SE2(), 0.0)

        if self._spin_cancel(resources, lambda: resources.motion.wait_until_complete(
                timeout=rospy.Duration.from_sec(0.1)), timeout_time):
            return CollectSampleResult(status=CollectSampleStatus.TIMEOUT)

        resources.motion.arm(False)

        return CollectSampleResult(status=CollectSampleStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()
