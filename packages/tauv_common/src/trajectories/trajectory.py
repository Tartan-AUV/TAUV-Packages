from abc import ABC, abstractmethod, abstractproperty
from typing import Any
from spatialmath import SE3, Twist3


class Trajectory(ABC):

    @abstractmethod
    def __init__(self,
                 start_pose: SE3, start_twist: Twist3,
                 end_pose: SE3, end_twist: Twist3,
                 params: Any):
        pass

    @abstractmethod
    def evaluate(self, time: float) -> (SE3, Twist3):
        pass

    @property
    @abstractmethod
    def duration(self) -> float:
        pass
