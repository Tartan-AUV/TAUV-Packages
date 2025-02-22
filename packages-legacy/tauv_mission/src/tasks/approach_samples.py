import rospy
import numpy as np
from spatialmath import SE3, SO3, SE2
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
import time
from tauv_util.spatialmath import flatten_se3
from math import atan2


class ApproachSamplesStatus(TaskStatus):
    SUCCESS = 0
    CANCELLED = 1
    TIMEOUT = 2
    SAMPLES_NOT_FOUND = 3

@dataclass
class ApproachSamplesResult(TaskResult):
    status: ApproachSamplesStatus


class ApproachSamples(Task):
    def __init__(self, sample_tags):
        super().__init__()
        self._sample_tags = sample_tags

    def _get_samples_mean(self, resources: TaskResources, initial=False):
        sample_detections = [resources.map.find_one(tag) for tag in self._sample_tags]
        valid_sample_detections = list(filter(lambda x: x is not None, sample_detections))

        if len(valid_sample_detections) == 0:
            if initial: print(">> !! No sample detections on initial survey found")
            return None

        odom_t_samples_array = [detection.pose for detection in valid_sample_detections]
        odom_r_samples_mean = np.mean(
            np.array([odom_t_sample.t for odom_t_sample in odom_t_samples_array]),
            axis=0
        )

        return odom_r_samples_mean

    def _create_overhead_r_from_mean(self, samples_mean):
        overhead_r = np.array([samples_mean[0], samples_mean[1], 0.7])
        return overhead_r

    def run(self, resources: TaskResources) -> ApproachSamplesResult:
        resources.motion.cancel()

        rate = 5 # Hz
        sleep_time_s = 1.0 / rate

        last_odom_r_samples_mean = None
        planar_error_thresh_flag = False
        initial_survey_flag = True

        timeout_time = rospy.Time.now() + rospy.Duration.from_sec(25)
        print('Approaching samples mean')

        """
        odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
        odom_r_samples_mean = self._get_samples_mean(resources, initial=True)
        
        if odom_r_samples_mean is None:
            return ApproachSamplesResult(status=ApproachSamplesStatus.SAMPLES_NOT_FOUND)
        
        odom_r_samples_mean_overhead = self._create_overhead_r_from_mean(odom_r_samples_mean)
        odom_t_goal = SE3.Rt(odom_t_vehicle.R, odom_r_samples_mean_overhead)
        resources.motion.goto(odom_t_goal)
        
        while True:
            if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                break

            if self._check_cancel(resources): return CircleBuoyResult(CircleBuoyStatus.TIMEOUT)
        
        return ApproachSamplesResult(status=ApproachSamplesStatus.SUCCESS)
        """

        while rospy.Time.now() < timeout_time and not planar_error_thresh_flag:
            odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
            odom_r_samples_mean = self._get_samples_mean(resources, initial=initial_survey_flag)

            if initial_survey_flag:
                initial_survey_flag = False

                if odom_r_samples_mean is None:
                    return ApproachSamplesResult(status=ApproachSamplesStatus.SAMPLES_NOT_FOUND)
                else:
                    last_odom_r_samples_mean = odom_r_samples_mean
            elif odom_r_samples_mean is not None:
                last_odom_r_samples_mean = odom_r_samples_mean

            odom_r_samples_mean_overhead = self._create_overhead_r_from_mean(last_odom_r_samples_mean)

            odom_t_goal = SE3.Rt(odom_t_vehicle.R, odom_r_samples_mean_overhead)
            resources.motion.goto(odom_t_goal)

            planar_error_thresh_flag = np.linalg.norm(odom_t_vehicle.t[:2] - odom_r_samples_mean_overhead[:2]) <= 0.3

            if self._check_cancel(resources): return ApproachSamplesResult(ApproachSamplesStatus.TIMEOUT)

            time.sleep(sleep_time_s)

        return ApproachSamplesResult(status=ApproachSamplesStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()
