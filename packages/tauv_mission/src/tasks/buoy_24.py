import rospy
import numpy as np
from spatialmath import SE3, SO3, SE2
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
import time
from tauv_util.spatialmath import flatten_se3
from math import atan2


class CircleBuoyStatus(TaskStatus):
    SUCCESS = 0
    CANCELLED = 1
    TIMEOUT = 2
    BUOY_NOT_FOUND = 3
    BUOY_LOST = 4
    ERROR = 5

@dataclass
class CircleBuoyResult(TaskResult):
    status: CircleBuoyStatus


class CircleBuoy(Task):
    def __init__(self, tag: str,
                 circle_radius: float=1.5,
                 circle_ccw=True,
                 waypoint_every_n_meters=0.5,
                 stare_timeout_s=8,
                 circle_depth=0.7,
                 latch_buoy=False):
        super().__init__()
        
        self._circle_radius = circle_radius
        self._circle_ccw = circle_ccw
        self._tag = tag
        self._stare_timeout_s = stare_timeout_s
        self._circle_depth = circle_depth
        self._latch_buoy = latch_buoy

        self._n_waypoints_along_circle_trajectory = int(2 * np.pi * self._circle_radius / waypoint_every_n_meters)

    def run(self, resources: TaskResources) -> CircleBuoyResult:
        # odom_t_course = resources.transforms.get_a_to_b('kf/odom', 'kf/course')
        
        try:
            resources.map.reset()
            print("Map reset for buoy_24 task")
        except rospy.ServiceException as e:
            print("Error calling map reset from buoy_24 task")
            return CircleBuoyResult(status=CircleBuoyStatus.ERROR)
        
        time.sleep(self._stare_timeout_s)
        
        # Try to detect buoy and approach the circling position
        odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
        buoy_detection = resources.map.find_closest(self._tag, position=odom_t_vehicle.t)

        if buoy_detection is None:
            print("Could not detect buoy after staring for {self._stare_timeout_s} s")
            return CircleBuoyResult(status=CircleBuoyStatus.BUOY_NOT_FOUND)
        
        vec_vehicle_to_bouy = buoy_detection.pose.t - odom_t_vehicle.t
        vec_vehicle_to_bouy_norm = np.linalg.norm(vec_vehicle_to_bouy)

        if vec_vehicle_to_bouy_norm > 10.0:
            print(f"Buoy detection found > 10m (real: {vec_vehicle_to_bouy_norm : .1f} m) away")
        
        vec_vehicle_to_bouy_hat = vec_vehicle_to_bouy / vec_vehicle_to_bouy_norm
        circling_point_start = buoy_detection.pose.t - self._circle_radius * vec_vehicle_to_bouy_hat
        circling_point_orientation = SO3.Rz(atan2(vec_vehicle_to_bouy[1], vec_vehicle_to_bouy[0]))
        odom_t_circling_point = SE3.Rt(circling_point_orientation, circling_point_start)
        
        print(f"Approach buoy from {vec_vehicle_to_bouy_norm}m away")
        resources.motion.goto(odom_t_circling_point, params=resources.motion.get_trajectory_params("rapid"))

        while True:
            if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                break

            if self._check_cancel(resources): return CircleBuoyResult(status=CircleBuoyStatus.CANCELLED)
        
        # Begin circling, updating the position of the buoy as we make more detections
        t_intersection = atan2(odom_t_vehicle.t[1] - buoy_detection.pose.t[1], odom_t_vehicle.t[0] - buoy_detection.pose.t[0])
        
        if self._circle_ccw:
            t_final = t_intersection - 2 * np.pi
        else:
            t_final = t_intersection + 2 * np.pi
        
        circle_fn = lambda t : np.array([buoy_detection.pose.t[0], buoy_detection.pose.t[1], self._circle_depth]) + \
                               np.array([self._circle_radius * np.cos(t), self._circle_radius * np.sin(t), 0])
        
        t_circle_eval_array = np.linspace(t_intersection, t_final, self._n_waypoints_along_circle_trajectory)
        circle_waypoints = np.array([circle_fn(t) for t in t_circle_eval_array])
        
        last_buoy_detection = buoy_detection
        cumulative_offset = np.zeros(3)
        
        for i, waypoint in enumerate(circle_waypoints):
            odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
            new_buoy_detection = resources.map.find_closest(self._tag, position=last_buoy_detection.pose.t)
            resources.transforms.set_a_to_b('kf/odom', 'buoy/center', new_buoy_detection.pose)
            
            if new_buoy_detection is not None:
                print(f"Buoy detections found. Updating buoy center for waypoints i > {i}/{len(circle_waypoints)}...")
                vec_last_to_new_detection = new_buoy_detection.pose.t - last_buoy_detection.pose.t
                vec_last_to_new_detection[2] = 0.0
                last_buoy_detection = new_buoy_detection
            else:
                vec_last_to_new_detection = np.zeros(3)
            
            # Update circle points and go to adjusted waypoint
            if not self._latch_buoy:
                cumulative_offset += vec_last_to_new_detection

            adjusted_waypoint = waypoint + cumulative_offset
            
            vec_vehicle_to_adjusted_waypoint = last_buoy_detection.pose.t - adjusted_waypoint
            adjusted_waypoint_orientation = SO3.Rz(atan2(vec_vehicle_to_adjusted_waypoint[1], vec_vehicle_to_adjusted_waypoint[0]))
            
            odom_t_adjusted_waypoint = SE3.Rt(adjusted_waypoint_orientation, adjusted_waypoint)
            resources.motion.goto(odom_t_adjusted_waypoint, params=resources.motion.get_trajectory_params("rapid"))

            while True:
                if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                    break

                if self._check_cancel(resources): return CircleBuoyResult(status=CircleBuoyStatus.CANCELLED)
            
        print("Finished circling buoy")
        return CircleBuoyResult(status=CircleBuoyStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()
