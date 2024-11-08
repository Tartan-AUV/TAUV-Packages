import rospy
import numpy as np
from spatialmath import SE3, SO3, SE2
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
import time
from tauv_util.spatialmath import flatten_se3
from math import atan2

class DebugDepthStatus(TaskStatus):
    SUCCESS = 0

@dataclass
class DebugDepthResult(TaskResult):
    status: DebugDepthStatus

class DebugDepth(Task):
    def __init__(self):
        super().__init__()

    def run(self, resources: TaskResources):
        odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')

        print ('finding closest to vehicle')
        for i in range(15):
            detection = resources.map.find_closest('buoy_24', odom_t_vehicle.t)
            print(f'i: {i}, {np.linalg.norm(detection.pose.t - odom_t_vehicle.t) - 0.476}')
            time.sleep(1)

        print ('find closest to last detection')
        for i in range(15):
            detection = resources.map.find_closest('buoy_24', detection.pose.t)
            print(f'i: {i}, {np.linalg.norm(detection.pose.t - odom_t_vehicle.t) - 0.476}')
            time.sleep(1)

        print('done')

        return DebugDepthResult(status=DebugDepthStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        pass
