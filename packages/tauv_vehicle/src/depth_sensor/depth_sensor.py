import rospy
from sensor_msgs.msg import Temperature as TemperatureMsg
from tauv_msgs.msg import FluidDepth as DepthMsg
from .ms5837lib import ms5837


class DepthSensor():
    def __init__(self):
        self._dt = 0.10
        self._depth_pub = rospy.Publisher('depth', DepthMsg, queue_size=10)
        self._temp_pub = rospy.Publisher('temperature', TemperatureMsg, queue_size=10)

        self._ms5837 = ms5837.MS5837_02BA(bus=8)

        while not self._init() and not rospy.is_shutdown():
            print('[depth_sensor] init failed, retrying')
            rospy.sleep(1)

        if rospy.is_shutdown():
            return

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _init(self) -> bool:
        try:
            self._ms5837.init()
            return True
        except (IOError, OSError) as e:
            print(f'[depth_sensor] _init: {e}')
            return False

    def _update(self, timer_event):
        try:
            self._ms5837.read()
        except (IOError, OSError) as e:
            print(f'[depth_sensor] _update: {e}')
            while not self._init() and not rospy.is_shutdown():
                print('[depth_sensor] init failed, retrying')
                rospy.sleep(1)
            return

        timestamp = rospy.Time.now()

        depth_msg = DepthMsg()
        depth_msg.header.stamp = timestamp
        depth_msg.header.frame_id = 'depth_sensor_link'
        depth_msg.depth = self._ms5837.depth()
        self._depth_pub.publish(depth_msg)

        temp_msg = TemperatureMsg()
        temp_msg.header.stamp = timestamp
        temp_msg.temperature = self._ms5837.temperature()
        self._temp_pub.publish(temp_msg)


def main():
    rospy.init_node('depth_sensor')
    d = DepthSensor()
    d.start()

