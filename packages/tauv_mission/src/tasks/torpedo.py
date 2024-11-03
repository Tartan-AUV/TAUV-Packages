import rospy
from spatialmath import SE3, SO3
from dataclasses import dataclass
from typing import Optional
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
from tauv_util.spatialmath import flatten_se3
import numpy as np
import time
from math import pi


class TorpedoStatus(TaskStatus):
    SUCCESS = 0
    CANCELLED = 1
    BUOY_NOT_FOUND = 3
    BUOY_LOST = 4


@dataclass
class TorpedoResult(TaskResult):
    status: TorpedoStatus


class Torpedo(Task):

    def __init__(self,
                 tag: str,
                 timeout: float,
                 approach_wait_duration: float, shoot_wait_duration: float,
                 approach_distance: float, shoot_distance: float,
                 approach_pos_error: float, approach_yaw_error: float,
                 shoot_pos_error: float, shoot_yaw_error: float,
                 shoot_torpedo: Optional[int] = None):
        super().__init__()

        self._tag = tag
        self._timeout = timeout
        self._approach_wait_duration = approach_wait_duration
        self._shoot_wait_duration = shoot_wait_duration
        self._approach_distance = approach_distance
        self._shoot_distance = shoot_distance
        self._approach_pos_error = approach_pos_error
        self._approach_yaw_error = approach_yaw_error
        self._shoot_pos_error = shoot_pos_error
        self._shoot_yaw_error = shoot_yaw_error
        self._shoot_torpedo = shoot_torpedo

    def run(self, resources: TaskResources) -> TorpedoResult:
        timeout_time = rospy.Time.now() + rospy.Duration.from_sec(self._timeout)

        odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
        buoy_detection = resources.map.find_closest(self._tag, odom_t_vehicle.t)

        vehicle_t_torpedo = resources.transforms.get_a_to_b('kf/vehicle', 'kf/torpedo')
        vehicle_t_cam = resources.transforms.get_a_to_b('kf/vehicle', 'kf/oakd_bottom')
        vehicle_t_cam_aligned = vehicle_t_cam * SE3(SO3.TwoVectors(x="z", y="x"))

        buoy_t_vehicle_shoot = SE3.Tx(-self._shoot_distance) * vehicle_t_torpedo.inv()
        buoy_t_vehicle_approach = SE3.Tx(-self._approach_distance) * vehicle_t_cam_aligned.inv()

        if buoy_detection is None:
            return TorpedoResult(status=TorpedoStatus.BUOY_NOT_FOUND)

        resources.motion.cancel()

        odom_t_buoy = buoy_detection.pose

        if np.linalg.norm(odom_t_buoy.t - odom_t_vehicle.t) > 5:
            return TorpedoResult(status=TorpedoStatus.BUOY_NOT_FOUND)

        approach_complete = False

        while rospy.Time.now() < timeout_time:
            odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
            buoy_detection = resources.map.find_closest(self._tag, odom_t_buoy.t)

            if buoy_detection is None:
                return TorpedoResult(status=TorpedoStatus.BUOY_LOST)

            odom_t_buoy = buoy_detection.pose
            odom_t_buoy = flatten_se3(odom_t_buoy)

            resources.transforms.set_a_to_b('kf/odom', 'buoy', odom_t_buoy)

            if not approach_complete:
                odom_t_vehicle_goal = odom_t_buoy * buoy_t_vehicle_approach
                resources.transforms.set_a_to_b('kf/odom', 'vehicle_goal', odom_t_vehicle_goal)
                resources.motion.goto(odom_t_vehicle_goal, params=resources.motion.get_trajectory_params("feedback"))

                # # Wait for motion to complete
                # while True:
                #     if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                #         break
                #     if self._check_cancel(resources): return TorpedoResult(status=TorpedoStatus.CANCELLED)

                # # Wait approach_wait_duration
                # wait_start_time = time.time()
                # while time.time() - wait_start_time < self._approach_wait_duration:
                #     if self._check_cancel(resources): return TorpedoResult(TorpedoStatus.CANCELLED)
                #     time.sleep(0.1)

                # Update poses
                # If close enough to approach position, go in to shoot
                pos_error = np.linalg.norm(odom_t_vehicle_goal.t - odom_t_vehicle.t)
                yaw_error = abs((pi + odom_t_vehicle_goal.rpy()[0] - odom_t_vehicle.rpy()[0]) % (2 * pi) - pi)

                print(f"approach pos error: {pos_error}")
                print(f"approach yaw error: {yaw_error}")

                if pos_error < self._approach_pos_error and yaw_error < self._approach_yaw_error:
                    approach_complete = True
            else:
                resources.transforms.set_a_to_b('kf/odom', 'buoy', odom_t_buoy)

                # Goto shoot position
                odom_t_vehicle_goal = odom_t_buoy * buoy_t_vehicle_shoot
                resources.transforms.set_a_to_b('kf/odom', 'vehicle_goal', odom_t_vehicle_goal)
                resources.motion.goto(odom_t_vehicle_goal, params=resources.motion.get_trajectory_params("feedback"))

                # # Wait for motion to complete
                # while True:
                #     if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                #         break
                #     if self._check_cancel(resources): return TorpedoResult(status=TorpedoStatus.CANCELLED)
                #

                pos_error = np.linalg.norm(odom_t_vehicle_goal.t - odom_t_vehicle.t)
                yaw_error = abs((pi + odom_t_vehicle_goal.rpy()[0] - odom_t_vehicle.rpy()[0]) % (2 * pi) - pi)

                if pos_error < self._shoot_pos_error and yaw_error < self._shoot_yaw_error:
                    # Wait shoot_wait_duration
                    wait_start_time = time.time()
                    while time.time() - wait_start_time < self._shoot_wait_duration:
                        if self._check_cancel(resources): return TorpedoResult(TorpedoStatus.CANCELLED)
                        time.sleep(0.1)

                    # Shoot
                    print("shoot!")
                    if self._shoot_torpedo is not None:
                        resources.actuators.shoot_torpedo(self._shoot_torpedo)

                    # Wait shoot_wait_duration
                    wait_start_time = time.time()
                    while time.time() - wait_start_time < self._shoot_wait_duration:
                        if self._check_cancel(resources): return TorpedoResult(TorpedoStatus.CANCELLED)
                        time.sleep(0.1)

                    # Finish
                    break

        resources.motion.goto_relative(SE3.Tx(-2))

        while True:
            if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                break

            if self._check_cancel(resources): return TorpedoResult(status=TorpedoStatus.CANCELLED)

        return TorpedoResult(status=TorpedoStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()