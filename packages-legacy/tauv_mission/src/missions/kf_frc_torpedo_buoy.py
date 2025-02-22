from typing import Optional
from spatialmath import SE3, SO3, SE2
from missions.mission import Mission
from tasks.task import Task, TaskResult
from tasks import dive, scan_rotate, hit_buoy, surface
from enum import IntEnum


class State(IntEnum):
    UNKNOWN = 0
    DIVE = 1
    SCAN = 2
    TORPEDO_1 = 3
    TORPEDO_2 = 4
    BUOY_1 = 5
    BUOY_2 = 6
    SURFACE = 7


class KFFRCTorpedoBuoy(Mission):

    def __init__(self):
        self._state = State.UNKNOWN

    def entrypoint(self) -> Optional[Task]:
        self._state = State.DIVE
        return dive.Dive(delay=5.0)

    def transition(self, task: Task, task_result: TaskResult) -> Optional[Task]:
        if self._state == State.DIVE \
                and task_result.status == dive.DiveStatus.SUCCESS:
            self._state = State.SCAN
            return scan_rotate.ScanRotate()
        elif self._state == State.SCAN \
                and task_result.status == scan_rotate.ScanRotateStatus.SUCCESS:
            self._state = State.TORPEDO_1
            return hit_buoy.HitBuoy(tag='torpedo_22_circle', timeout=100, frequency=10, distance=0.3, error_a=1, error_b=0.1, error_threshold=0.1, shoot_torpedo=0)
        elif self._state == State.TORPEDO_1 \
                and task_result.status == hit_buoy.HitBuoyStatus.SUCCESS:
            self._state = State.TORPEDO_2
            return hit_buoy.HitBuoy(tag='torpedo_22_trapezoid', timeout=100, frequency=10, distance=0.3, error_a=1, error_b=0.1, error_threshold=0.1, shoot_torpedo=1)
        elif self._state == State.TORPEDO_2 \
                and task_result.status == hit_buoy.HitBuoyStatus.SUCCESS:
            self._state = State.BUOY_1
            return hit_buoy.HitBuoy(tag='buoy_23_earth_1', timeout=100, frequency=10, distance=0.3, error_a=1, error_b=0.1, error_threshold=0.1, shoot_torpedo=None)
        elif self._state == State.BUOY_1 \
                and task_result.status == hit_buoy.HitBuoyStatus.SUCCESS:
            self._state = State.BUOY_2
            return hit_buoy.HitBuoy(tag='buoy_23_earth_2', timeout=100, frequency=10, distance=0.3, error_a=1, error_b=0.1, error_threshold=0.1, shoot_torpedo=None)
        elif self._state == State.BUOY_2 \
                and task_result.status == hit_buoy.HitBuoyStatus.SUCCESS:
            self._state = State.SURFACE
            return surface.Surface()
        else:
            return None
