import rospy
import numpy as np
from spatialmath import SE3, SO3
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
import time

class PickChevronStatus(TaskStatus):
    SUCCESS = 0
    FAILURE = 1
    CHEVRON_NOT_FOUND = 2
    CHEVRON_LOST = 3

@dataclass
class PickChevronResult(TaskResult):
    status: PickChevronStatus


class PickChevron(Task):

    def __init__(self):
        super().__init__()

    def run(self, resources: TaskResources) -> PickChevronResult:
        odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
        vehicle_t_suction = resources.transforms.get_a_to_b('kf/vehicle', 'kf/suction')
        detection = resources.map.find_closest('chevron', odom_t_vehicle.t)
        # if len(detections) == 0:
        #     return PickChevronResult(PickChevronStatus.CHEVRON_NOT_FOUND)
        odom_t_chevron = detection.pose

        chevron_t_suction = SE3.Rt(SO3.TwoVectors(x='x', z='-z'), np.array([0, 0, 0]))

        if odom_t_chevron is None:
            return PickChevronResult(status=PickChevronStatus.CHEVRON_NOT_FOUND)

        resources.actuators.move_arm(1.0)
        resources.actuators.activate_suction(1.0)

        while True:
            odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
            detection = resources.map.find_closest('chevron', odom_t_vehicle.t)
            odom_t_chevron = detection.pose
            odom_t_chevron = SE3.Rt(SO3.RPY((0, 0, odom_t_chevron.rpy()[2])), odom_t_chevron.t)

            goal_odom_t_vehicle = odom_t_chevron * chevron_t_suction * vehicle_t_suction.inv()
            resources.transforms.set_a_to_b('kf/odom', 'kf/goal', goal_odom_t_vehicle)
            resources.transforms.set_a_to_b('kf/odom', 'chevron', odom_t_chevron)
            resources.transforms.set_a_to_b('kf/odom', 'suction', odom_t_chevron * chevron_t_suction)


            if np.linalg.norm(goal_odom_t_vehicle.t - odom_t_vehicle.t) + abs(goal_odom_t_vehicle.rpy()[2] - odom_t_vehicle.rpy()[2]) < 0.1:
                break

            error_factor = 10

            xy_error = np.linalg.norm(goal_odom_t_vehicle.t[0:2] - odom_t_vehicle.t[0:2]) + abs(goal_odom_t_vehicle.rpy()[2] - odom_t_vehicle.rpy()[2])
            z = odom_t_vehicle.t[2] + np.exp(-error_factor * xy_error) * (goal_odom_t_vehicle.t[2] - odom_t_vehicle.t[2])

            target_odom_t_vehicle = SE3.Rt(goal_odom_t_vehicle.R, np.array([
                goal_odom_t_vehicle.t[0],
                goal_odom_t_vehicle.t[1],
                z
            ]))

            resources.transforms.set_a_to_b('kf/odom', 'kf/target', target_odom_t_vehicle)

            resources.motion.goto(target_odom_t_vehicle)

            if self._check_cancel(resources): return PickChevronResult(PickChevronStatus.FAILURE)

            time.sleep(0.1)

        resources.motion.goto_relative(SE3.Rt(SO3(), np.array([0, 0, -0.2])))

        return PickChevronResult(status=PickChevronStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.actuators.activate_suction(0.0)
        resources.actuators.move_arm(0.0)
        resources.motion.cancel()
