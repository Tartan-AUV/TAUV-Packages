import genesis as gs
from spatialmath import SE3, SO3

from tauv_sim.vehicle import Vehicle

class World:

    def __init__(self):
        self.pool_mesh = "/home/gleb/dev/tauv-mono/packages-legacy/tauv_sim_worlds/models/umd/meshes/umd.obj"

    def construct_scene(self, headless):
        self.scene = gs.Scene(
            show_viewer=not headless,
            # renderer = gs.renderers.RayTracer(),
        )
        self.pool = self.scene.add_entity(gs.morphs.Mesh(
                                                file=self.pool_mesh,
                                                fixed=True,
                                                euler=(0.0, 90.0, 0.0)))

        self.scene.build()
        for i in range(1000):
           self.scene.step()

    def add_vehicle(self, vehicle: Vehicle, pose: SE3):
        pass
