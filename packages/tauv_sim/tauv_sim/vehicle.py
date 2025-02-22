import genesis as gs

from abc import ABC, abstractmethod
from pathlib import Path

class Vehicle(ABC):

    def __init__(self, frame_name: str):
        self.frame_name = frame_name

    @abstractmethod
    def get_mesh_file(self) -> str:
        pass


class Kingfisher(Vehicle):

    def __init__(self,
                 frame_name="kf"):
        super().__init__(frame_name)
        self._mesh_path = Path("/home/gleb/dev/tauv-mono/packages-legacy/tauv_config/kingfisher_sim_description/meshes/KingfisherCoarseNoInsides.obj")

    def get_mesh_file(self) -> str:
        return str(self._mesh_path)

