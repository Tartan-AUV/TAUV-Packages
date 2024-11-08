import random
import time

import rospy
from std_msgs.msg import Float64
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse


class Accumulator:

    def __init__(self):
        self._i: float = 0.0

        self._sub: rospy.Subscriber = rospy.Subscriber("acc/in", Float64, self._handle_in)
        self._pub: rospy.Publisher = rospy.Publisher("acc/out", Float64, queue_size=10)
        self._srv: rospy.Service = rospy.Service("acc/reset", Trigger, self._handle_reset)

    def _handle_in(self, msg: Float64):
        x = msg.data
        self._i = self._i + x
        print(f"i is now {self._i}")

        out_msg = Float64()
        out_msg.data = self._i
        self._pub.publish(out_msg)

    def _handle_reset(self, req: TriggerRequest) -> TriggerResponse:
        res = TriggerResponse()
        if random.randint(0, 10) < 9:
            res.success = True
            res.message = "Success"
        else:
            res.success = False
            res.message = "Unlucky"

        self._i = 0


def main():
    a = Accumulator()
    rospy.spin()

# NEW FILE

class RandomNumberGenerator():

    def __init__(self):
        self._pub: rospy.Publisher = rospy.Publisher("acc/in", Float64, queue_size=10)

        self._timer: rospy.Timer = rospy.Timer(rospy.Duration(1), self._handle_timer)

    def _handle_timer(self, _: rospy.TimerEvent):
        x = random.randint(5, 12)
        msg = Float64()
        msg.data = x
        self._pub.publish(msg)

    def _handle_stop(self, msg: Trigger):
        self._timer.shutdown()


def main():
    r = RandomNumberGenerator()
    rospy.spin()

