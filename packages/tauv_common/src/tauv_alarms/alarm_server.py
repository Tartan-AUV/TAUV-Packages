import rospy

from tauv_alarms.alarm_util import AlarmType, FailureLevel
from tauv_alarms.alarms import Alarm
from tauv_msgs.msg import AlarmReport, AlarmWithMessage, ReadableAlarmReport
from tauv_msgs.srv import SyncAlarms
from threading import Lock
import typing
from tauv_messages.messager import Messager

class AlarmServer:
    def __init__(self):
        self._active: typing.Set[AlarmType] = set()
        self._stamp: rospy.Time = rospy.Time.now()
        self._lock = Lock()
        self._logger = Messager('Alarm', color=130)

        for at in Alarm:
            if at.default_set:
                self._active.add(at)

        self.pub = rospy.Publisher('alarms/report', AlarmReport, queue_size=10)
        self.pub_readable = rospy.Publisher('alarms/readable', ReadableAlarmReport, queue_size=10)
        rospy.Service('alarms/sync', SyncAlarms, self.handle_request)
        rospy.Timer(rospy.Duration(0.1), self.pub_report)
        rospy.Timer(rospy.Duration(0.5), self.print_readable)

    def pub_report(self, timer_event):
        report = AlarmReport()
        with self._lock:
            report.stamp = rospy.Time.now()
            report.active_alarms = [a.id for a in self._active]
        self.pub.publish(report)

    def handle_request(self, srv: SyncAlarms._request_class):
        res: typing.List[AlarmWithMessage] = SyncAlarms._response_class()
        with self._lock:
            self._stamp = rospy.Time.now()
            for a in srv.diff:
                if a.set:
                    if Alarm(a.id) not in self._active:
                        self._logger.log(f"{Alarm(a.id).name} set: {a.message}", severity=Messager.SEV_WARNING)
                    self._active.add(Alarm(a.id))
                else:
                    if Alarm(a.id) in self._active:
                        self._logger.log(f"{Alarm(a.id).name} cleared: {a.message}", severity=Messager.SEV_INFO)
                    self._active.discard(Alarm(a.id))
            res.active_alarms = [a.id for a in self._active]
            res.stamp = self._stamp
            res.success = True
        return res

    def print_readable(self, timer_event):
        rep = ReadableAlarmReport()
        maxfl = FailureLevel.NO_FAILURE
        for a in self._active:
            if a.failure_level == FailureLevel.NO_FAILURE:
                rep.alarms_no_failure.append(a.name)
            elif a.failure_level == FailureLevel.PREDIVE_FAILURE:
                rep.alarms_predive_failure.append(a.name)
            elif a.failure_level == FailureLevel.MISSION_FAILURE:
                rep.alarms_mission_failure.append(a.name)
            elif a.failure_level == FailureLevel.CRITICAL_FAILURE:
                rep.alarms_critical_failure.append(a.name)
            
            if a.failure_level.value > maxfl.value:
                maxfl = a.failure_level
        rep.failure_level = maxfl.name
        self.pub_readable.publish(rep)

def main():
    rospy.init_node('alarm_server')
    AlarmServer()
    rospy.spin()