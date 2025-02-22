from math import pi
from dataclasses import dataclass
import rospy
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
from spatialmath import SE3
import time

class YawRollStatus(TaskStatus):
    SUCCESS = 0
    FAILURE = 1

@dataclass
class YawRollResult(TaskResult):
    status: YawRollStatus

class YawRoll(Task):
    def __init__(self):
        super().__init__()

    def run(self, resources: TaskResources) -> YawRollResult:
        odom_t_vehicle_initial = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle');

        for i in range(1, 6):
            theta = i * (4 * pi) / 5

            resources.motion.goto(odom_t_vehicle_initial * SE3.Rz(theta))
            while True:
                if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                    break
                
                if self._check_cancel(resources):
                    return YawRollResult(YawRollStatus.FAILURE)

            time.sleep(1.0)

        return YawRollResult(YawRollStatus.SUCCESS)
    
    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()
