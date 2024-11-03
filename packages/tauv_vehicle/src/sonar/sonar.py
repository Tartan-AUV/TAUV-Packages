from .ping_python.brping.ping360 import Ping360
from .ping_python.brping import definitions

from tauv_msgs.msg import SonarPulse as PulseMsg
from std_msgs.msg import Header

import math

import rospy

RAD_PER_GRAD = math.pi / 200.0


class Sonar:
    def __init__(self):
        self._dt = 1.0 / rospy.get_param('sonar/device/frequency')

        port = rospy.get_param('sonar/device/port')
        baudrate = rospy.get_param('sonar/device/baudrate')

        self._ping = Ping360()
        self._ping.connect_serial(port, baudrate)

        while not self._ping.initialize() and not rospy.is_shutdown():
            print(f'[sonar] ping360 init failed')
            rospy.sleep(1.0)

        self._gain_setting = rospy.get_param('sonar/device/gain')
        self._transmit_duration = rospy.get_param('sonar/device/transmit_duration')
        self._range = rospy.get_param('sonar/device/range')
        self._transmit_frequency = rospy.get_param('sonar/device/transmit_frequency')
        self._num_samples = rospy.get_param('sonar/device/num_samples')
        self._sonar_link = rospy.get_param('sonar/device/sonar_link', 'sonar_link')

        self._sample_period = self._calculate_sample_period(self._range, self._num_samples, 1500)

        self._pulse_pub = rospy.Publisher('pulse', PulseMsg, queue_size=10)

        self._update_config()

        self._angular_speed_grad_per_s = 1000.0 * 400.0 / 2400.0
        self._angle = 0

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _update(self, timer_event):
        self._angle = (1 + self._angle) % 400
        self._do_pulse_at_angle(self._angle)

        res = self._ping.wait_message([definitions.COMMON_NACK, definitions.PING360_DEVICE_DATA], 0.1)

        if res is None:
            return

        if res.message_id == definitions.PING360_DEVICE_DATA:
            pulse_msg = self._get_pulse_msg(res)
            self._pulse_pub.publish(pulse_msg)
        elif res.message_id == definitions.COMMON_NACK:
            print(f'[sonar] _update: error communicating with ping360 device: {res.nack_message}')

    def _do_pulse_at_angle(self, angle):
        self._ping.transmitAngleNoWait(angle)

    def _calculate_sample_period(self, distance, num_samples, speed_sound, sample_period_tick_duration=25e-9):
        return int(2 * distance / (num_samples * speed_sound * sample_period_tick_duration))

    def _get_pulse_msg(self, pinger_data) -> PulseMsg:
        msg = PulseMsg()
        msg.header = Header()
        msg.header.frame_id = self._sonar_link
        msg.header.stamp = rospy.Time.now()

        msg.mode = pinger_data.mode
        msg.gain_setting = pinger_data.gain_setting
        msg.angle = pinger_data.angle # * RAD_PER_GRAD
        msg.transmit_duration = pinger_data.transmit_duration
        msg.sample_period = pinger_data.sample_period
        msg.transmit_frequency = pinger_data.transmit_frequency
        msg.number_of_samples = pinger_data.number_of_samples
        msg.data_length = pinger_data.data_length
        msg.data = [b for b in bytearray(pinger_data.data)]

        return msg

    def _update_config(self):
        self._ping.set_gain_setting(self._gain_setting)
        self._ping.set_transmit_frequency(self._transmit_frequency)
        self._ping.set_transmit_duration(self._transmit_duration)
        self._ping.set_sample_period(self._sample_period)
        self._ping.set_number_of_samples(self._num_samples)


def main():
    rospy.init_node('sonar')
    s = Sonar()
    s.start()
