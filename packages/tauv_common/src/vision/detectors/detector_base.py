from abc import ABC, abstractmethod


class Detector(ABC):
    def __init__(self, name, param):
        self.name = name
        self.param = param

    @abstractmethod
    def get_detection(self, image):
        pass
