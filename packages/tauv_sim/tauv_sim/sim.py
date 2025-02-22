import rclpy
from rclpy.node import Node
from tauv_sim.world import World
import genesis as gs


class Sim(Node):
    def __init__(self):
        super().__init__('sim')
        self.headless = False
        self.backend = gs.cpu
        gs.init(self.backend)
        self.world = World()
        self.world.construct_scene(headless=False)

def main():
    rclpy.init()
    node = Sim()

if __name__ == '__main__':
    main()
