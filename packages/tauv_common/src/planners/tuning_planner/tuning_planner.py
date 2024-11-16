import rospy
import numpy as np
from enum import Enum
from typing import Dict, Optional

from geometry_msgs.msg import Vector3, Wrench
from sensor_msgs.msg import Joy as JoyMsg
from std_srvs.srv import SetBool


class ButtonInput(Enum):
    ESTOP = 'estop'
    ARM = 'arm'
    TRIGGER = 'trigger'


class AxisInput(Enum):
    X = 'x'
    Y = 'y'
    Z = 'z'
    ROLL = 'roll'
    PITCH = 'pitch'
    YAW = 'yaw'


class TuningPlanner:
    def __init__(self):
        self._dt: float = 0.05
        self._command_timeout: float = 0.1

        self._is_armed: bool = False

        self._joy_sub: rospy.Subscriber = rospy.Subscriber('joy', JoyMsg, self._handle_joy)
        self._wrench_pub: rospy.Publisher = rospy.Publisher('wrench', Wrench, queue_size=10)
        self._arm_srv: rospy.ServiceProxy = rospy.ServiceProxy('arm', SetBool)

        self._command: np.array = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._command_timestamp: rospy.Time = rospy.Time.now()

        self._trigger_timeout: float = 5.0
        self._trigger_command: Optional[np.array] = None
        self._trigger_timestamp: Optional[rospy.Time] = None
        self._wrench_index: int = 0

        self._load_config()

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _update(self, timer_event):
        update_time = rospy.Time.now()

        command_timeout: bool = (update_time - self._command_timestamp).to_sec() > self._command_timeout
        trigger_timeout: bool = self._trigger_timestamp is None or (update_time - self._trigger_timestamp).to_sec() > self._trigger_timeout

        if trigger_timeout:
            self._trigger_command = None
            self._trigger_timestamp = None

        if not self._is_armed or command_timeout:
            self._command = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        if self._is_armed and not trigger_timeout:
            command = self._trigger_command
        else:
            command = self._command

        wrench: Wrench = Wrench()
        wrench.force = Vector3(command[0], command[1], command[2])
        wrench.torque = Vector3(command[3], command[4], command[5])
        self._wrench_pub.publish(wrench)

    def _handle_joy(self, msg: JoyMsg):
        command_time = rospy.Time.now()

        if self._is_pressed(msg, ButtonInput.ESTOP):
            print('ESTOP')
            self._arm(False)
        elif self._is_pressed(msg, ButtonInput.ARM):
            print('ARM')
            self._arm(True)

        if self._is_pressed(msg, ButtonInput.TRIGGER) and self._trigger_timestamp is None:
            self._trigger(command_time)

        self._command = np.array([
            self._axis_value(msg, AxisInput.X),
            self._axis_value(msg, AxisInput.Y),
            self._axis_value(msg, AxisInput.Z),
            self._axis_value(msg, AxisInput.ROLL),
            self._axis_value(msg, AxisInput.PITCH),
            self._axis_value(msg, AxisInput.YAW)
        ])
        print(self._command)
        self._command_timestamp = command_time

    def _trigger(self, time: rospy.Time):
        self._trigger_timestamp = time
        self._trigger_command = self._wrenches[self._wrench_index]
        print(self._trigger_command)
        self._wrench_index = (self._wrench_index + 1) % len(self._wrenches)

    def _arm(self, arm: bool):
        self._is_armed = arm
        try:
            resp = self._arm_srv(arm)
            if not resp.success:
                rospy.logwarn('Arm request failed')
        except rospy.ServiceException:
            rospy.logwarn('Arm server not responding')

    def _axis_value(self, msg: JoyMsg, axis: AxisInput):
        return self._axis_scales[axis.value] * msg.axes[self._axis_ids[axis.value]]

    def _is_pressed(self, msg: JoyMsg, button: ButtonInput):
        return msg.buttons[self._button_ids[button.value]] == 1

    def _load_config(self):
        self._button_ids: Dict[str, int] = rospy.get_param('~button_ids')
        self._axis_ids: Dict[str, int] = rospy.get_param('~axis_ids')
        self._axis_scales: Dict[str, int] = rospy.get_param('~axis_scales')
        self._wrenches: [np.array] = list(map(
            lambda l: np.array(l),
            rospy.get_param('~wrenches')
        ))


def main():
    rospy.init_node('tuning_planner')
    p = TuningPlanner()
    p.start()