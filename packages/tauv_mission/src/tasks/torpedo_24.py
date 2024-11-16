import rospy
import numpy as np
from spatialmath import SE3, SO3
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
from tauv_util.spatialmath import flatten_se3
from typing import Optional
from scipy.optimize import least_squares
from math import pi
from enum import Enum


class Torpedo24Status(TaskStatus):
    SUCCESS = 0
    CANCELLED = 1
    TIMEOUT = 2
    BUOY_NOT_FOUND = 3
    BUOY_LOST = 4

@dataclass
class Torpedo24Result(TaskResult):
    status: Torpedo24Status


class Torpedo24Target(Enum):
    TOP_RIGHT = 0
    TOP_LEFT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3


@dataclass
class Torpedo24Solution:
    odom_t_banner: SE3
    odom_t_top_right: SE3
    odom_t_top_left: SE3
    odom_t_bottom_left: SE3
    odom_t_bottom_right: SE3

    def get_target(self, target: Torpedo24Target) -> SE3:
        if target == Torpedo24Target.TOP_RIGHT:
            return self.odom_t_top_right
        elif target == Torpedo24Target.TOP_LEFT:
            return self.odom_t_top_left
        elif target == Torpedo24Target.BOTTOM_LEFT:
            return self.odom_t_bottom_left
        else:
            return self.odom_t_bottom_right


class Torpedo24(Task):

    def __init__(self, timeout: float, frequency: float, dry_run: bool):
        super().__init__()

        self._timeout = timeout
        self._period = rospy.Duration.from_sec(1 / frequency)
        self._dry_run = dry_run

        self._distance = 0.3
        self._error_a = 0.2
        self._error_b = 1.0
        self._error_threshold = 0.1
        self._approach_distance = 2.5


    def run(self, resources: TaskResources) -> Torpedo24Result:
        timeout_time = rospy.Time.now() + rospy.Duration.from_sec(self._timeout)

        result = self._run_single(resources, timeout_time, 1, Torpedo24Target.TOP_LEFT)
        if result.status != Torpedo24Status.SUCCESS:
            return result

        result = self._run_single(resources, timeout_time, 0, Torpedo24Target.BOTTOM_RIGHT)
        if result.status != Torpedo24Status.SUCCESS:
            return result

        return Torpedo24Result(status=Torpedo24Status.SUCCESS)

    def _run_single(self, resources: TaskResources, timeout_time: rospy.Time, torpedo: int, target: Torpedo24Target) -> Torpedo24Result:

        vehicle_t_torpedo = resources.transforms.get_a_to_b('kf/vehicle', f'kf/torpedo_{torpedo}')
        target_t_vehicle_goal = SE3.Tx(-self._distance) * vehicle_t_torpedo.inv()

        while rospy.Time.now() < timeout_time:
            odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')

            solution = self._find_solution(resources)

            if solution is None:
                print("no solution!")
                return Torpedo24Result(status=Torpedo24Status.BUOY_NOT_FOUND)

            odom_t_target = solution.get_target(target)

            resources.transforms.set_a_to_b('kf/odom', 'solution_banner', solution.odom_t_banner)
            resources.transforms.set_a_to_b('kf/odom', 'solution_top_right', solution.odom_t_top_right)
            resources.transforms.set_a_to_b('kf/odom', 'solution_top_left', solution.odom_t_top_left)
            resources.transforms.set_a_to_b('kf/odom', 'solution_bottom_left', solution.odom_t_bottom_left)
            resources.transforms.set_a_to_b('kf/odom', 'solution_bottom_right', solution.odom_t_bottom_right)

            odom_t_vehicle_approach = solution.odom_t_banner * SE3.Tx(-self._approach_distance)

            resources.transforms.set_a_to_b('kf/odom', 'approach', odom_t_vehicle_approach)

            if np.linalg.norm(odom_t_vehicle.t - odom_t_vehicle_approach.t) < 0.5:
                break

            resources.motion.goto(odom_t_vehicle_approach, params=resources.motion.get_trajectory_params("feedback"))

            if self._check_cancel(resources): return Torpedo24Result(status=Torpedo24Status.CANCELLED)

            rospy.sleep(self._period)

        while rospy.Time.now() < timeout_time:
            odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
            target_detection = resources.map.find_closest("torpedo_24_octagon", odom_t_target.t)

            if target_detection is None:
                print("lost target!")
                return Torpedo24Result(status=Torpedo24Status.BUOY_LOST)

            odom_t_target = target_detection.pose
            odom_t_target = flatten_se3(odom_t_target)
            odom_t_vehicle_goal = odom_t_target * target_t_vehicle_goal

            if np.linalg.norm(odom_t_vehicle.t - odom_t_vehicle_goal.t) < self._error_threshold:
                break

            target_t_vehicle = odom_t_target.inv() * odom_t_vehicle

            orthogonal_error = np.linalg.norm(target_t_vehicle.t[1:3] - target_t_vehicle_goal.t[1:3]) + abs(target_t_vehicle.rpy()[2] - target_t_vehicle_goal.rpy()[2])

            x = -self._error_a * (1 - np.exp(-self._error_b * orthogonal_error))

            target_t_vehicle_target = SE3.Rt(SO3(), (x + target_t_vehicle_goal.t[0], target_t_vehicle_goal.t[1], target_t_vehicle_goal.t[2]))

            odom_t_vehicle_target = odom_t_target * target_t_vehicle_target

            resources.transforms.set_a_to_b('kf/odom', 'target', odom_t_target)
            resources.transforms.set_a_to_b('kf/odom', 'vehicle_goal', odom_t_vehicle_goal)
            resources.transforms.set_a_to_b('kf/odom', 'vehicle_target', odom_t_vehicle_target)

            resources.motion.goto(odom_t_vehicle_target, params=resources.motion.get_trajectory_params("feedback"))

            if self._check_cancel(resources): return Torpedo24Result(status=Torpedo24Status.CANCELLED)

            rospy.sleep(self._period)

        rospy.sleep(5.0)

        print("shoot!")
        if not self._dry_run:
            for _ in range(5):
                resources.actuators.shoot_torpedo(torpedo)
                rospy.sleep(1.0)

        resources.motion.goto_relative(SE3.Rt(SO3(), (-self._approach_distance, 0, 0)))

        while True:
            if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                break

            if self._check_cancel(resources): return Torpedo24Result(status=Torpedo24Status.CANCELLED)

        return Torpedo24Result(status=Torpedo24Status.SUCCESS)

    def _find_solution(self, resources: TaskResources) -> Optional[Torpedo24Solution]:
        odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')

        buoy_detections = resources.map.find('torpedo_24')
        target_detections = resources.map.find('torpedo_24_octagon')

        if buoy_detections is None or target_detections is None:
            return None

        def distance(detection):
            return np.linalg.norm(detection.pose.t - odom_t_vehicle.t)

        buoy_detections = sorted(buoy_detections, key=distance)

        print(f"buoy detections: {buoy_detections}")
        print(f"target detections: {target_detections}")

        for buoy_detection in buoy_detections:
            print(f"buoy detection: {buoy_detection}")

            close_target_detections = list(filter(lambda detection: np.linalg.norm(detection.pose.t - buoy_detection.pose.t) < 1.5, target_detections))

            print(f"close target detections: {close_target_detections}")

            if len(close_target_detections) < 4:
                continue

            xy_positions = np.array([
                buoy_detection.pose.t[0:2]
            ] + [
                target_detection.pose.t[0:2]
                for target_detection in close_target_detections
            ])

            yaw = fit_yaw_angle(xy_positions, look_yaw=buoy_detection.pose.rpy()[2])

            odom_t_banner = SE3.Rt(
                SO3.RPY((0, 0, yaw)),
                buoy_detection.pose.t,
            )

            odom_t_targets = [detection.pose for detection in close_target_detections]
            banner_t_targets = [odom_t_banner.inv() * odom_t_target for odom_t_target in odom_t_targets]

            banner_t_targets_t = np.array([
                banner_t_target.t
                for banner_t_target in banner_t_targets
            ])

            banner_t_target_angles = np.arctan2(-banner_t_targets_t[:, 2], banner_t_targets_t[:, 1]) % (2 * pi)

            odom_t_targets = [
                pair[1] for pair in list(sorted(zip(banner_t_target_angles, odom_t_targets)))
            ]

            solution = Torpedo24Solution(
                odom_t_banner=odom_t_banner,
                odom_t_top_right=odom_t_targets[0],
                odom_t_top_left=odom_t_targets[1],
                odom_t_bottom_left=odom_t_targets[2],
                odom_t_bottom_right=odom_t_targets[3],
            )

            return solution

        return None

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()


def fit_yaw_angle(xy_positions: np.array, look_yaw: float) -> float:
    x = xy_positions[:, 0]
    y = xy_positions[:, 1]

    m, b = np.polyfit(x, y, 1)

    yaw = (np.arctan(m) + pi / 2) % (2 * pi)

    # Make sure yaw points away from us, not towards us
    if np.abs(yaw - look_yaw) > pi / 2:
        yaw += pi

    return yaw

