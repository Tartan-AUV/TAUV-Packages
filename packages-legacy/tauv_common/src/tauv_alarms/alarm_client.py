import rospy
import typing

from tauv_alarms.alarm_util import AlarmType, FailureLevel
from tauv_alarms.alarms import Alarm
from tauv_msgs.msg import AlarmReport, AlarmWithMessage
from tauv_msgs.srv import SyncAlarms
from threading import Lock

class AlarmClient:
    def __init__(self, monitor=False) -> None:
        self._monitor = monitor
        self._lastupdated = rospy.Time(0)

        self._active: typing.Set[AlarmType] = set()
        self._active.add(Alarm.UNKNOWN_ALARMS) # start with unknown alarms until we hear from server.
        
        self._timeout = rospy.Duration(1)   # 1 second timeout
        self._lock = Lock()

        # connect to server if we're not in monitor mode.
        rospy.Subscriber('alarms/report', AlarmReport, self._update_report)
        self.pub = rospy.Publisher('alarms/post', AlarmReport, queue_size=10)
        self.sync = rospy.ServiceProxy('alarms/sync', SyncAlarms)

        if not monitor:
            # wait for first report from the server
            while not self._lastupdated > rospy.Time(0):
                rospy.sleep(0.05)

    def _update_report(self, msg: AlarmReport):
        if msg.stamp > self._lastupdated:
            with self._lock:
                self._lastupdated = msg.stamp
                self._active = set([Alarm(i) for i in msg.active_alarms])
    
    def set(self, a: AlarmType, msg: str="", value: bool=True):
        if self._monitor:
            raise RuntimeError("Alarm Clients in Monitor mode cannot set/clear exceptions!")

        diff = None
        with self._lock:
            if value and a not in self._active:
                self._active.add(a)
                self._lastupdated = rospy.Time.now()
                diff = AlarmWithMessage()
                diff.id = a.id
                diff.message = msg
                diff.set = True
            elif not value and a in self._active:
                self._active.remove(a)
                self._lastupdated = rospy.Time.now()
                diff = AlarmWithMessage()
                diff.id = a.id
                diff.message = msg
                diff.set = False
        
        if diff is not None:
            req = SyncAlarms._request_class()
            req.diff = [diff]
            
            try:
                res: SyncAlarms._response_class = self.sync(req)
            except rospy.ServiceException as e:
                raise RuntimeWarning(f"Failed to set exception! {e}")
            
            if not res.success:
                raise RuntimeWarning(f"Failed to set exception! Server returned an error.")
            
            with self._lock:
                self._lastupdated = res.stamp
                self._active = set([Alarm(i) for i in res.active_alarms])
    
    def clear(self, a: AlarmType, msg=""):
        self.set(a, msg, False)

    def get_active_alarms(self, include_whitelisted=False):
        if rospy.Time.now() - self._lastupdated > self._timeout:
            return self._active.union(set([Alarm.UNKNOWN_ALARMS]))
        return self._active

    def check(self, a: AlarmType) -> bool:
        return a in self._active or (a == Alarm.UNKNOWN_ALARMS and rospy.Time.now() - self._lastupdated > self._timeout)

    def get_failure_level(self) -> FailureLevel:
        fl = FailureLevel.NO_FAILURE
        for a in self._active:
            if a.failure_level.value > fl.value:
                fl = a.failure_level
        return fl