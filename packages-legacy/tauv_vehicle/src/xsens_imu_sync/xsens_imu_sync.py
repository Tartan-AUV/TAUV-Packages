import rospy
import numpy as np

from tauv_msgs.msg import XsensImuSync as ImuSyncMsg, XsensImuData as ImuDataMsg
from std_msgs.msg import Header

from tauv_alarms import Alarm, AlarmClient

class ImuSync:
    VAR_OFFSET_INIT = 2e-3 ** 2
    VAR_SKEW_INIT = 1e-3 ** 2
    VAR_OFFSET = 2e-3 ** 2
    VAR_SKEW = 2e-6 ** 2

    def __init__(self):
        self._ac: AlarmClient = AlarmClient()

        self._sync_pub = rospy.Publisher('vehicle/xsens_imu/sync', ImuSyncMsg, queue_size=10)
        self._data_pub = rospy.Publisher('vehicle/xsens_imu/data', ImuDataMsg, queue_size=10)
        self._raw_data_sub = rospy.Subscriber('vehicle/xsens_imu/raw_data', ImuDataMsg, self._handle_imu_data)

        self._x = np.array([0, 0])

        self._P = np.array([[ImuSync.VAR_OFFSET_INIT, 0], [0, ImuSync.VAR_SKEW_INIT]])

        self._H = np.array([[1, 0], [0, 0]])

        self._Q = np.array([[0, 0], [0, ImuSync.VAR_SKEW]])

        self._R = ImuSync.VAR_OFFSET

        self._sync_period = rospy.Duration.from_sec(rospy.get_param('~sync_period'))
        self._expected_sync_time = None

        self._last_corrected_time = None
        self._last_ros_time = None
        self._last_imu_time = None


    def start(self):
        rospy.spin()

    def _handle_imu_data(self, data: ImuDataMsg):
        ros_time = data.ros_time
        imu_time = data.imu_time

        if self._last_imu_time is None:
            self._last_imu_time = imu_time
            self._x = np.array([(ros_time - imu_time).to_sec(), 0])
    
        dt = (imu_time - self._last_imu_time).to_sec()

        F = np.array([[1, dt], [0, 1]])
        self._x = np.matmul(F, self._x)

        self._P = np.matmul(F, np.matmul(self._P, np.transpose(F))) + dt * self._Q

        S = np.matmul(self._H, np.matmul(self._P, np.transpose(self._H))) + self._R

        K = np.matmul(self._P, np.transpose(self._H)) * (1 / S)

        residual = np.array([(ros_time - imu_time).to_sec(), 0]) - np.matmul(self._H, self._x)

        self._x = self._x + np.matmul(K, residual)
        self._P = np.matmul(np.identity(2) - np.matmul(K, self._H), self._P)

        corrected_time = self.convert_imu_time(imu_time)

        if data.triggered_dvl:
            self._expected_sync_time = corrected_time + self._sync_period
        elif self._expected_sync_time is not None and corrected_time > self._expected_sync_time:
            self._publish_missed_sync(self._expected_sync_time)
            self._expected_sync_time = self._expected_sync_time + self._sync_period

        msg = ImuSyncMsg()
        msg.header = Header()
        msg.header.stamp = corrected_time
        msg.ros_time = ros_time
        msg.imu_time = imu_time
        msg.triggered_dvl = data.triggered_dvl

        if not self._last_corrected_time is None:
            msg.d_corrected = (corrected_time - self._last_corrected_time).to_sec()

        if not self._last_ros_time is None:
            msg.d_ros = (ros_time - self._last_ros_time).to_sec()

        if not self._last_imu_time is None:
            msg.d_imu = (imu_time - self._last_imu_time).to_sec()

        self._sync_pub.publish(msg)

        data_msg = ImuDataMsg()
        data_msg.header = Header()
        data_msg.header.stamp = corrected_time
        data_msg.ros_time = ros_time
        data_msg.imu_time = imu_time
        data_msg.triggered_dvl = data.triggered_dvl
        data_msg.orientation = data.orientation
        data_msg.rate_of_turn = data.rate_of_turn
        data_msg.linear_acceleration = data.linear_acceleration
        data_msg.free_acceleration = data.free_acceleration

        self._data_pub.publish(data_msg)

        self._last_corrected_time = corrected_time
        self._last_ros_time = ros_time
        self._last_imu_time = imu_time

        self._ac.clear(Alarm.IMU_SYNC_NOT_INITIALIZED)
        self._ac.clear(Alarm.IMU_NOT_INITIALIZED)

    def convert_imu_time(self, imu_time):
        dt = (imu_time - self._last_imu_time).to_sec()

        converted_secs = imu_time.to_sec() + self._x[0] + dt * self._x[1]
        return rospy.Time.from_sec(converted_secs)

    def _publish_missed_sync(self, time):
        msg = ImuSyncMsg()
        msg.header = Header()
        msg.header.stamp = time
        msg.triggered_dvl = True
        msg.d_corrected = self._sync_period.to_sec()

def main():
    rospy.init_node('imu_sync')
    n = ImuSync()
    n.start()
