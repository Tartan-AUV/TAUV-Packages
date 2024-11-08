import rospy

from tauv_msgs.msg import Message

class Messager:
    SEV_ERROR=0
    SEV_WARNING=1
    SEV_INFO=2
    SEV_DEBUG=3

    # color: ANSI 256 color code: https://www.lihaoyi.com/post/BuildyourownCommandLinewithANSIescapecodes.html#256-colors
    def __init__(self, name, color=15) -> None:
        self._code = color
        self._pub = rospy.Publisher('messages', Message, queue_size=10)
        self._name = name

    def log(self, message, severity=SEV_INFO):
        m = Message()
        m.stamp = rospy.Time.now()
        m.message = f"[ {self._name} ] {message}"
        m.color_code_256 = self._code
        m.severity = severity
        self._pub.publish(m)

        if severity == Messager.SEV_DEBUG:
            rospy.logdebug(message)
        elif severity == Messager.SEV_INFO:
            rospy.loginfo(message)
        elif severity == Messager.SEV_WARNING:
            rospy.logwarn(message)
        elif severity == Messager.SEV_ERROR:
            rospy.logerr(message)