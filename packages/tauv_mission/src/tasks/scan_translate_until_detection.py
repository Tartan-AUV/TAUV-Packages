import time
import rospy
from spatialmath import SE2, SE3, SO3
import numpy as np
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
from typing import List

class ScanTranslateUntilDetectionStatus(TaskStatus):
    SUCCESS = 0
    FAILURE = 1


@dataclass
class ScanTranslateUntilDetectionResult(TaskResult):
    status: ScanTranslateUntilDetectionStatus


class ScanTranslateUntilDetection(Task):
    def __init__(self, course_t_start: SE3, points: [(float, float)], tags: List[str]):
        # Scan translate until any tag in tags is detected or we exhaust waypoints
        super().__init__()

        self._course_t_start = course_t_start
        self._points = points
        
        if not isinstance(tags, list):
            self._tags = [tags]
        else:
            self._tags = tags

    def run(self, resources: TaskResources) -> ScanTranslateUntilDetectionResult:
        odom_t_course = resources.transforms.get_a_to_b('kf/odom', 'kf/course')
        odom_t_start = odom_t_course * self._course_t_start

        for (y, z) in self._points:
            start_t_vehicle_goal = SE3.Rt(SO3(), (0, y, z))
            odom_t_vehicle_goal = odom_t_start * start_t_vehicle_goal

            resources.motion.goto(odom_t_vehicle_goal)

            while True:
                # Check for detection of the tag
                tag_detection = [ resources.map.find_one(tag) for tag in self._tags ]
                valid_detections = [ detection for detection in tag_detection if detection is not None ]
                
                if len(valid_detections) > 0:
                    detected_tags = ", ".join([ detection.tag for detection in valid_detections ])

                    print(f"Detected a tag ({len(valid_detections)}/{len(self._tags)} found)")
                    print(f"\tTags detected: {detected_tags}")
                    
                    return ScanTranslateUntilDetectionResult(ScanTranslateUntilDetectionStatus.SUCCESS)

                if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                    break

                if self._check_cancel(resources): return ScanTranslateUntilDetectionResult(ScanTranslateUntilDetectionStatus.FAILURE)

        return ScanTranslateUntilDetectionResult(ScanTranslateUntilDetectionStatus.FAILURE)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()