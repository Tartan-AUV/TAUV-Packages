# This node prints the messages to screen in a pretty way, (and tries
# to grab any messages already printed?)
import rospy
from tauv_msgs.msg import Message
from tauv_messages.messager import Messager

def c(color: int):
    return f"\u001b[38;5;{color}m"
r="\u001b\u001b[0m"

ec = {
    Messager.SEV_ERROR: 196,
    Messager.SEV_WARNING: 172,
    Messager.SEV_DEBUG: 59,
    Messager.SEV_INFO: 15
}
es = {
    Messager.SEV_ERROR:   "[ERROR]",
    Messager.SEV_WARNING: "[ WARN]",
    Messager.SEV_DEBUG:   "[DEBUG]",
    Messager.SEV_INFO:    "[ INFO]"
}

class MessagePrinter:

    def __init__(self) -> None:
        self.sub = rospy.Subscriber("messages", Message, self.handle_message)

    def handle_message(self, msg: Message):
        s = msg.severity
        m = msg.message
        code = msg.color_code_256
        t = msg.stamp.to_sec()
        
        print(f"[{t:.1f}]{ec[s]}{es[s]}{r} {c(code)}{m}{r}")
        
        # if s == Messager.SEV_DEBUG:
        #     rospy.logdebug(m)
        # elif s == Messager.SEV_INFO:
        #     rospy.loginfo(m)
        # elif s == Messager.SEV_WARNING:
        #     rospy.logwarn(m)
        # elif s == Messager.SEV_ERROR:
        #     rospy.logerr(m)

def main():
    rospy.init_node('message_printer')
    MessagePrinter()
    rospy.spin()