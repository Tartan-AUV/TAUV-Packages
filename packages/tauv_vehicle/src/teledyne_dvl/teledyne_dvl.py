import rospy
from typing import List

from tauv_msgs.msg import XsensImuSync as ImuSyncMsg, TeledyneDvlData as DvlDataMsg

from .pathfinder import Pathfinder

from tauv_alarms import Alarm, AlarmClient

class TeledyneDVL:
    def __init__(self):
        self._ac = AlarmClient()

        self._data_pub: rospy.Publisher = rospy.Publisher('vehicle/teledyne_dvl/data', DvlDataMsg, queue_size=10)
        self._sync_sub: rospy.Subscriber = rospy.Subscriber('vehicle/xsens_imu/sync', ImuSyncMsg, self._handle_sync)

        self._sync_timestamps: List[rospy.Time] = []

        port = rospy.get_param('~port')
        baudrate = rospy.get_param('~baudrate')

        self._pf: Pathfinder = Pathfinder(port, baudrate)

    def start(self):
        self._pf.open()
        # self._pf.configure()
        self._pf.start_measuring()

        self._ac.clear(Alarm.DVL_NOT_INITIALIZED)

        n_missed = 0

        while not rospy.is_shutdown():
            ensemble = self._pf.poll()

            if ensemble is None:
                rospy.logwarn_throttle(10, 'No ensemble')
                n_missed += 1
            else:
                n_missed = 0

            if n_missed > 5:
                rospy.logwarn("Attempting reset")
                self._pf.close()
                self._pf.open()
                self._pf.start_measuring()

            if ensemble is None:
                continue

            rospy.logdebug(f'[teledyne_dvl] timestamps: {list(map(lambda t: t.to_sec(), self._sync_timestamps))}')

            self._sweep_sync_timestamps(ensemble.receive_time)

            valid_timestamps = list(filter(
                lambda t: (ensemble.receive_time - t).to_sec() >= Pathfinder.MIN_MEASURE_TIME,
                self._sync_timestamps
            ))

            rospy.logdebug(f'[teledyne_dvl] swept_timestamps: {list(map(lambda t: t.to_sec(), self._sync_timestamps))}')
            rospy.logdebug(f'[teledyne_dvl] valid_timestamps: {list(map(lambda t: t.to_sec(), valid_timestamps))}')

            msg: DvlDataMsg = ensemble.to_msg()

            if msg.is_hr_velocity_valid:
                self._ac.clear(Alarm.DVL_NO_LOCK)
            else:
                self._ac.set(Alarm.DVL_NO_LOCK)

            if len(valid_timestamps) > 0:
                self._ac.clear(Alarm.DVL_NO_TIMESTAMPS)
                msg.header.stamp = valid_timestamps[0] + rospy.Duration.from_sec(sum(msg.beam_time_to_bottoms) / 4)
                self._sync_timestamps = list(filter(
                    lambda t: t.to_sec() != valid_timestamps[0].to_sec(),
                   self._sync_timestamps
                ))
                rospy.logdebug(f'[teledyne_dvl] new valid_timestamps: {list(map(lambda t: t.to_sec(), self._sync_timestamps))}')
            else:
                rospy.logwarn(f'[teledyne_dvl] no valid timestamps')
                self._ac.set(Alarm.DVL_NO_TIMESTAMPS)

            self._data_pub.publish(msg)

        self._pf.close()

    def _handle_sync(self, data: ImuSyncMsg):
        time: rospy.Time = data.header.stamp

        if data.triggered_dvl:
            self._sync_timestamps.append(time)

    def _sweep_sync_timestamps(self, time: rospy.Time):
        self._sync_timestamps = list(filter(
                lambda t: (time - t).to_sec() < Pathfinder.MAX_MEASURE_TIME,
                self._sync_timestamps
        ))


def main():
    rospy.init_node('teledyne_dvl')
    n = TeledyneDVL()
    n.start()
