from socket import timeout
import rospy
from tauv_alarms.alarm_client import Alarm, AlarmClient

KickableDeadlines = {
    Alarm.CAMERA_NO_VIDEO_FRONT: rospy.Time(0),
    Alarm.CAMERA_NO_VIDEO_BOTTOM: rospy.Time(0),
    Alarm.GNC_NO_POSE: rospy.Time(0),
    Alarm.BUCKET_LIST_NOT_PUBLISHING: rospy.Time(0),
    Alarm.DARKNET_NOT_PUBLISHING: rospy.Time(0)
}

class Watchdog:
    def __init__(self) -> None:
        self.ac = AlarmClient()

        self.cam_frontsub = rospy.Subscriber("zedm_A/zed_node_A/left/camera_info", rospy.AnyMsg, lambda m: self.kick(m, Alarm.CAMERA_NO_VIDEO_FRONT, 1.0))
        self.cam_bottomsub = rospy.Subscriber("zedm_B/zed_node_B/left/camera_info", rospy.AnyMsg, lambda m: self.kick(m, Alarm.CAMERA_NO_VIDEO_BOTTOM, 1.0))
        self.gncsub = rospy.Subscriber("gnc/state_estimation/navigation_state", rospy.AnyMsg, lambda m: self.kick(m, Alarm.GNC_NO_POSE, 1.0))
        self.bucket_sub = rospy.Subscriber("bucket_list", rospy.AnyMsg, lambda m: self.kick(m, Alarm.BUCKET_LIST_NOT_PUBLISHING, 5.0))
        self.darknet_sub = rospy.Subscriber("darknet_ros/check_for_objects/status", rospy.AnyMsg, lambda m: self.kick(m, Alarm.DARKNET_NOT_PUBLISHING, 1.0))

        rospy.Timer(rospy.Duration(0.02), self.check)

    def check(self, timer_event):
        n = rospy.Time.now()
        for a in KickableDeadlines.keys():
            self.ac.set(a, "set by watchdog", n > KickableDeadlines[a])

    def kick(self, msg, a: Alarm, timeout: float = 1.0):
        KickableDeadlines[a] = rospy.Time.now() + rospy.Duration(timeout)

def main():
    rospy.init_node('watchdog')
    Watchdog()
    rospy.spin()