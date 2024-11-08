import rospy
from abc import ABC, abstractmethod
from enum import IntEnum
from dataclasses import dataclass
from threading import Event
from motion_client import MotionClient
from actuator_client import ActuatorClient
from map_client import MapClient
from transform_client import TransformClient
from typing import Callable, Any


@dataclass
class TaskResources:
    motion: MotionClient
    actuators: ActuatorClient
    map: MapClient
    transforms: TransformClient


TaskStatus = IntEnum


@dataclass
class TaskResult:
    status: TaskStatus


class Task(ABC):

    def __init__(self) -> None:
        self._cancel_event: Event = Event()
        self._cancel_complete_event: Event = Event()

    @abstractmethod
    def run(self, resources: TaskResources) -> TaskResult:
        pass

    @abstractmethod
    def _handle_cancel(self, resources: TaskResources):
        pass

    def cancel(self):
        rospy.logdebug('cancel setting cancel_event')
        self._cancel_event.set()
        self._cancel_complete_event.wait()

    def _spin_cancel(self, resources: TaskResources, func: Callable[[], bool], stop_time: rospy.Time) -> bool:
        while rospy.Time.now() < stop_time:
            if func():
                return False

            if self._check_cancel(resources): return True

        return True

    def _check_cancel(self, resources: TaskResources) -> bool:
        if self._cancel_event.is_set():
            self._cancel_event.clear()

            self._handle_cancel(resources)

            self._cancel_complete_event.set()

            return True

        return False
