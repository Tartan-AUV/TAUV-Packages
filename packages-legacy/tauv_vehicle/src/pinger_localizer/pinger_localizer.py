import rospy
import numpy as np
import uuid
import time

from std_msgs.msg import Float64, Float64MultiArray, Header
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from tauv_util.transforms import rpy_to_quat
from tauv_msgs.msg import PingDetection

from .backends.adalm_chained import ADALMChainedBackend
from .frontends.as1_rev_4 import AS1Rev4Frontend
from .processing.get_ping_frequency import get_ping_frequency
from .processing.remap_channels import remap_channels
from .processing.get_delays import get_delays_xcorr, get_delays_fft
from .processing.get_direction import get_direction
from .processing.filter_samples import filter_samples


class PingerLocalizer:

    def __init__(self):
        self._load_config()

        self._frontend: AS1Rev4Frontend = AS1Rev4Frontend(
            gain_pins=self._frontend_gain_pins,
        )

        self._backend: ADALMChainedBackend = ADALMChainedBackend(
            sample_frequency=self._backend_sample_frequency,
            sample_size=self._backend_sample_size,
            timeout=self._backend_timeout,
            source_id=self._backend_source_id,
            sink_id=self._backend_sink_id,
            source_trig_level=self._backend_source_trig_level,
            sink_trig_level=self._backend_sink_trig_level,
        )

        self._sample_pubs: [rospy.Publisher] = []
        for i in range(self._n_channels):
            sample_pub = rospy.Publisher(f'vehicle/pinger_localizer/sample/{i}', Float64MultiArray, queue_size=10)
            self._sample_pubs.append(sample_pub)

        self._ping_frequency_pub: rospy.Publisher = rospy.Publisher(f'vehicle/pinger_localizer/ping_frequency', Float64, queue_size=10)
        self._direction_pub: rospy.Publisher = rospy.Publisher(f'vehicle/pinger_localizer/direction', Vector3Stamped, queue_size=10)
        self._direction_pose_pub: rospy.Publisher = rospy.Publisher(f'vehicle/pinger_localizer/direction_pose', PoseStamped, queue_size=10)

        self._detection_pub: rospy.Publisher = rospy.Publisher(f'vehicle/pinger_localizer/detection', PingDetection, queue_size=10)

    def start(self):
        self._frontend.open()

        while not self._backend.open() and not rospy.is_shutdown():
            rospy.logerr('Open failed, trying again.')
            rospy.sleep(1.0)

        if rospy.is_shutdown():
            self._frontend.close()
            self._backend.close()
            return

        rospy.loginfo('Open succeeded.')

        while not rospy.is_shutdown():
            self._run()
            rospy.sleep(1.0)

        self._frontend.close()
        self._backend.close()

    def _run(self):
        self._frontend.set_gain(self._gain)

        sample_success, sample_times, samples = self._backend.sample()

        if not sample_success:
            rospy.loginfo('Sample failed.')
            return

        sample_times, samples = remap_channels(self._channel_mappings, sample_times, samples)

        # file_id = time.strftime("%Y%m%d-%H%M%S")
        # np.save(f'/data/pinger_localizer/right/{file_id}-times.npy', sample_times)
        # np.save(f'/data/pinger_localizer/right/{file_id}-samples.npy', samples)

        amplitudes = np.abs(samples)
        max_amplitudes = np.max(amplitudes, axis=1)
        rospy.loginfo(f'Max amplitude: {np.argmax(max_amplitudes)}')

        ping_frequency = get_ping_frequency(sample_times, samples, self._backend_sample_frequency, self._min_ping_frequency, self._max_ping_frequency)

        rospy.loginfo(f'Ping frequency: {ping_frequency} Hz')

        if abs(ping_frequency - 25000) < 2000:
            return

        self._ping_frequency_pub.publish(ping_frequency)

        for i in range(self._n_channels):
            msg = Float64MultiArray()
            msg.data = samples[i]
            self._sample_pubs[i].publish(msg)

        sample_times, samples = filter_samples(sample_times, samples, self._backend_sample_frequency, ping_frequency)

        delays = get_delays_xcorr(sample_times, samples, self._backend_sample_frequency, max_delay=self._max_delay)
        # delays = get_delays_fft(sample_times, samples, self._backend_sample_frequency, self._max_delay, self._interpolation_factor)

        rospy.loginfo(f'Delays: {delays}')

        direction, direction_psi, direction_theta = get_direction(delays, self._channel_positions)

        rospy.loginfo(f'Direction: {direction}')
        rospy.loginfo(f'Psi: {direction_psi}, Theta: {direction_theta}')

        curr_ros_time = rospy.Time.now()

        direction_msg = Vector3Stamped()
        direction_msg.header.frame_id = 'kf/vehicle'
        direction_msg.header.stamp = curr_ros_time
        direction_msg.x = direction[0]
        direction_msg.y = direction[1]
        direction_msg.z = direction[2]
        self._direction_pub.publish(direction_msg)

        direction_pose_msg = PoseStamped()
        direction_pose_msg.header.frame_id = 'kf/vehicle'
        direction_pose_msg.header.stamp = curr_ros_time
        direction_rpy = np.array([0, direction_theta, direction_psi])
        direction_quat = rpy_to_quat(direction_rpy)
        direction_pose_msg.pose.orientation = direction_quat
        self._direction_pose_pub.publish(direction_pose_msg)

        detection_msg = PingDetection()
        detection_msg.header.frame_id = 'kf/vehicle'
        detection_msg.header.stamp = curr_ros_time
        detection_msg.frequency = ping_frequency
        detection_msg.direction.x = direction[0]
        detection_msg.direction.y = direction[1]
        detection_msg.direction.z = direction[2]
        self._detection_pub.publish(detection_msg)

    def _load_config(self):
        self._n_channels = 4
        self._gain: int = rospy.get_param('~gain')
        self._channel_positions: np.array = np.array(rospy.get_param('~channel_positions'))
        self._channel_mappings: [int] = rospy.get_param('~channel_mappings')
        self._interpolation_factor: int = int(rospy.get_param('~interpolation_factor'))
        self._max_delay: float = rospy.get_param('~max_delay')
        self._min_ping_frequency: float = rospy.get_param('~min_ping_frequency')
        self._max_ping_frequency: float = rospy.get_param('~max_ping_frequency')
        self._frontend_gain_pins: [int] = rospy.get_param('~frontend/gain_pins')
        self._backend_sample_frequency: int = int(rospy.get_param('~backend/sample_frequency'))
        self._backend_sample_size: int = int(rospy.get_param('~backend/sample_size'))
        self._backend_timeout: float = rospy.get_param('~backend/timeout')
        self._backend_source_id: str = rospy.get_param('~backend/source_id')
        self._backend_sink_id: str = rospy.get_param('~backend/sink_id')
        self._backend_source_trig_level: float = rospy.get_param('~backend/source_trig_level')
        self._backend_sink_trig_level: float = rospy.get_param('~backend/sink_trig_level')


def main():
    rospy.init_node('pinger_localizer')
    n = PingerLocalizer()
    n.start()