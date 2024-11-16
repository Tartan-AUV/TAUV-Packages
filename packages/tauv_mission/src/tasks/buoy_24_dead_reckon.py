import rospy
import numpy as np
from spatialmath import SE3, SO3, SE2
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
import time
from tauv_util.spatialmath import flatten_se3
from math import atan2


class CircleBuoyDeadReckonStatus(TaskStatus):
    SUCCESS = 0
    CANCELLED = 1
    TIMEOUT = 2
    ERROR = 5

@dataclass
class CircleBuoyDeadReckonResult(TaskResult):
    status: CircleBuoyDeadReckonStatus


class CircleBuoyDeadReckon(Task):
    def __init__(self, course_t_circle_center,
                 circle_radius: float=1.5,
                 circle_ccw=True,
                 waypoint_every_n_meters=0.5,
                 circle_depth=0.7,
                 n_torpedos=0):
        super().__init__()
        
        self._circle_radius = circle_radius
        self._circle_ccw = circle_ccw
        self._circle_depth = circle_depth
        self._n_torpedos = n_torpedos

        self._course_t_circle_center = course_t_circle_center
        
        # self._n_waypoints_along_circle_trajectory = int(2 * np.pi * self._circle_radius / waypoint_every_n_meters)
        self._n_waypoints_along_circle_trajectory = 4

    def run(self, resources: TaskResources) -> CircleBuoyDeadReckonResult:
        # Try to detect buoy and approach the circling position
        odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
        odom_t_course = resources.transforms.get_a_to_b('kf/odom', 'kf/course')

        odom_t_circle_center = odom_t_course * self._course_t_circle_center

        vec_vehicle_to_buoy = odom_t_circle_center.t - odom_t_vehicle.t
        vec_vehicle_to_buoy_norm = np.linalg.norm(vec_vehicle_to_buoy)

        if vec_vehicle_to_buoy_norm > 10.0:
            print(f"We think buoy is > 10m (real: {vec_vehicle_to_buoy_norm : .1f} m) away")
        
        vec_vehicle_to_buoy_hat = vec_vehicle_to_buoy / vec_vehicle_to_buoy_norm
        circling_point_start = odom_t_circle_center.t - self._circle_radius * vec_vehicle_to_buoy_hat
        circling_point_orientation = SO3.Rz(atan2(vec_vehicle_to_buoy[1], vec_vehicle_to_buoy[0]))
        odom_t_circling_point = SE3.Rt(circling_point_orientation, circling_point_start)
        
        print(f"Approach buoy from {vec_vehicle_to_buoy_norm :.2f}m away")
        resources.motion.goto(odom_t_circling_point, params=resources.motion.get_trajectory_params("rapid"))

        while True:
            if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                break

            if self._check_cancel(resources): return CircleBuoyDeadReckonResult(status=CircleBuoyDeadReckonStatus.CANCELLED)
        
        # Begin circling, updating the position of the buoy as we make more detections
        t_intersection = atan2(odom_t_vehicle.t[1] - odom_t_circle_center.t[1], odom_t_vehicle.t[0] - odom_t_circle_center.t[0])
        
        if self._circle_ccw:
            t_final = t_intersection - 2 * np.pi
        else:
            t_final = t_intersection + 2 * np.pi
        
        circle_fn = lambda t : np.array([odom_t_circle_center.t[0], odom_t_circle_center.t[1], self._circle_depth]) + \
                               np.array([self._circle_radius * np.cos(t), self._circle_radius * np.sin(t), 0])
        
        t_circle_eval_array = np.linspace(t_intersection, t_final, self._n_waypoints_along_circle_trajectory)
        circle_waypoints = np.array([circle_fn(t) for t in t_circle_eval_array])

        circle_center_with_my_depth = np.array([odom_t_circle_center.t[0], odom_t_circle_center.t[1], self._circle_depth])
        circle_waypoints = [
            circle_center_with_my_depth + (odom_t_course * np.array([self._circle_radius, self._circle_radius, 0])).flatten(),
            circle_center_with_my_depth + (odom_t_course * np.array([self._circle_radius, -self._circle_radius, 0])).flatten(),
            circle_center_with_my_depth + (odom_t_course * np.array([-self._circle_radius, -self._circle_radius, 0])).flatten(),
            circle_center_with_my_depth + (odom_t_course * np.array([-self._circle_radius, self._circle_radius, 0])).flatten(),
        ]

        min_i = 0
        min_dist = float('inf')

        for i in range(4):
            waypoint_norm = np.linalg.norm(odom_t_vehicle.t - circle_waypoints[i])
            if waypoint_norm < min_dist:
                min_dist = waypoint_norm
                min_i = i

        course_R_goal = SO3()
        odom_R_goal = odom_t_course.R * course_R_goal

        for j in range(5):
            idx = (min_i + j) % 4
            waypoint = circle_waypoints[idx]
    
            odom_t_waypoint = SE3.Rt(odom_R_goal, waypoint)

            resources.motion.goto(odom_t_waypoint, params=resources.motion.get_trajectory_params("rapid"))

            while True:
                if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                    break

                if self._check_cancel(resources): return CircleBuoyDeadReckonResult(status=CircleBuoyDeadReckonStatus.CANCELLED)


        print("Finished circling buoy")
        return CircleBuoyDeadReckonResult(status=CircleBuoyDeadReckonStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()
