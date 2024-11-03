import rospy
from enum import Enum
from typing import Dict, Optional

from tauv_msgs.msg import ControllerCommand
from sensor_msgs.msg import Joy
from std_srvs.srv import SetBool, SetBoolResponse


class ButtonInput(Enum):
    ESTOP = 'estop'
    ARM = 'arm'
    MODE_FORCE = 'mode_force'
    MODE_AUTO = 'mode_auto'
    HOLD_ROLL = 'hold_roll'
    HOLD_PITCH = 'hold_pitch'
    HOLD_Z = 'hold_z'

class Mode(Enum):
    FORCE = 'force'
    AUTO = 'auto'

class AxisInput(Enum):
    X = 'x'
    Y = 'y'
    Z = 'z'
    ROLL = 'roll'
    PITCH = 'pitch'
    YAW = 'yaw'


class TeleopPlanner:
    def __init__(self):
        self._dt: float = 0.05

        self._parse_config()

        self._is_armed: bool = False
        self._mode: Mode = Mode.AUTO

        self._joy_cmd: Optional[ControllerCommand] = None
        self._joy_cmd_timestamp: Optional[rospy.Time] = None
        self._joy_cmd_timeout: float = 1.0

        self._planner_cmd: Optional[ControllerCommand] = None
        self._planner_cmd_timestamp: Optional[rospy.Time] = None
        self._planner_cmd_timeout: float = 1.0

        self._hold_z_pressed = False
        self._hold_roll_pressed = False
        self._hold_pitch_pressed = False
        self._hold_z = False
        self._hold_roll = False
        self._hold_pitch = False
        self._setpoint_z = 0.0
        self._setpoint_roll = 0.0
        self._setpoint_pitch = 0.0

        self._joy_sub: rospy.Subscriber = rospy.Subscriber('joy', Joy, self._handle_joy)
        self._planner_cmd_sub: rospy.Subscriber = rospy.Subscriber('gnc/planner_command', ControllerCommand, self._handle_planner_cmd)
        self._cmd_pub: rospy.Publisher = rospy.Publisher('gnc/controller_command', ControllerCommand, queue_size=10)
        self._arm_srv: rospy.ServiceProxy = rospy.ServiceProxy('vehicle/thrusters/arm', SetBool)

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _update(self, timer_event):
        time = rospy.Time.now()

        is_joy_cmd_valid = self._joy_cmd is not None \
            and self._joy_cmd_timestamp is not None \
            and (time - self._joy_cmd_timestamp).to_sec() < self._joy_cmd_timeout

        is_planner_cmd_valid = self._planner_cmd is not None \
            and self._planner_cmd_timestamp is not None \
            and (time - self._planner_cmd_timestamp).to_sec() < self._planner_cmd_timeout

        cmd = ControllerCommand()
        cmd.use_f_x = True
        cmd.use_f_y = True
        cmd.use_f_z = True
        cmd.use_f_roll = True
        cmd.use_f_pitch = True
        cmd.use_f_yaw = True

        if self._mode == Mode.AUTO and is_planner_cmd_valid:
            cmd = self._planner_cmd
        elif self._mode == Mode.FORCE and is_joy_cmd_valid:
            cmd = self._joy_cmd

        self._cmd_pub.publish(cmd)

    def _handle_joy(self, msg: Joy):
        if self._is_pressed(msg, ButtonInput.ESTOP):
            self._set_arm(False)
        elif self._is_pressed(msg, ButtonInput.ARM):
            self._set_arm(True)

        if self._is_pressed(msg, ButtonInput.MODE_FORCE):
            self._mode = Mode.FORCE
        elif self._is_pressed(msg, ButtonInput.MODE_AUTO):
            self._mode = Mode.AUTO

        if not self._hold_z_pressed and self._is_pressed(msg, ButtonInput.HOLD_Z):
            self._hold_z = not self._hold_z
        if not self._hold_roll_pressed and self._is_pressed(msg, ButtonInput.HOLD_ROLL):
            self._hold_roll = not self._hold_roll
        if not self._hold_pitch_pressed and self._is_pressed(msg, ButtonInput.HOLD_PITCH):
            self._hold_pitch = not self._hold_pitch

        self._hold_z_pressed = self._is_pressed(msg, ButtonInput.HOLD_Z)
        self._hold_roll_pressed = self._is_pressed(msg, ButtonInput.HOLD_ROLL)
        self._hold_pitch_pressed = self._is_pressed(msg, ButtonInput.HOLD_PITCH)

        if self._hold_z:
            self._setpoint_z = self._setpoint_z + self._axis_setpoint_value(msg, AxisInput.Z)
        if self._hold_roll:
            self._setpoint_roll = self._setpoint_roll + self._axis_setpoint_value(msg, AxisInput.ROLL)
        if self._hold_pitch:
            self._setpoint_pitch = self._setpoint_pitch + self._axis_setpoint_value(msg, AxisInput.PITCH)

        joy_cmd = ControllerCommand()
        joy_cmd.f_x = self._axis_force_value(msg, AxisInput.X)
        joy_cmd.f_y = self._axis_force_value(msg, AxisInput.Y)
        joy_cmd.f_z = self._axis_force_value(msg, AxisInput.Z) if not self._hold_z else 0
        joy_cmd.f_roll = self._axis_force_value(msg, AxisInput.ROLL) if not self._hold_roll else 0
        joy_cmd.f_pitch = self._axis_force_value(msg, AxisInput.PITCH) if not self._hold_pitch else 0
        joy_cmd.f_yaw = self._axis_force_value(msg, AxisInput.YAW)
        joy_cmd.use_f_x = True
        joy_cmd.use_f_y = True
        joy_cmd.use_f_z = not self._hold_z
        joy_cmd.use_f_roll = not self._hold_roll
        joy_cmd.use_f_pitch = not self._hold_pitch
        joy_cmd.use_f_yaw = True
        joy_cmd.setpoint_z = self._setpoint_z
        joy_cmd.setpoint_roll = self._setpoint_roll
        joy_cmd.setpoint_pitch = self._setpoint_pitch
        joy_cmd.use_setpoint_z = self._hold_z
        joy_cmd.use_setpoint_roll = self._hold_roll
        joy_cmd.use_setpoint_pitch = self._hold_pitch

        self._joy_cmd = joy_cmd
        self._joy_cmd_timestamp = rospy.Time.now()

    def _handle_planner_cmd(self, msg: ControllerCommand):
        self._planner_cmd = msg
        self._planner_cmd_timestamp = rospy.Time.now()

    def _set_arm(self, arm: bool):
        try:
            res: SetBoolResponse = self._arm_srv.call(arm)
            if not res.success:
                rospy.logwarn('Arm request failed')
            else:
                self._is_armed = arm
        except rospy.ServiceException:
            rospy.logwarn('Arm server not responding')

    def _axis_force_value(self, msg: Joy, axis: AxisInput):
        return self._axis_force_scales[axis.value] * msg.axes[self._axis_ids[axis.value]]

    def _axis_setpoint_value(self, msg: Joy, axis: AxisInput):
        return self._axis_setpoint_scales[axis.value] * msg.axes[self._axis_ids[axis.value]]

    def _is_pressed(self, msg: Joy, button: ButtonInput):
        return msg.buttons[self._button_ids[button.value]] == 1

    def _parse_config(self):
        self._button_ids: Dict[str, int] = rospy.get_param('~button_ids')
        self._axis_ids: Dict[str, int] = rospy.get_param('~axis_ids')
        self._axis_force_scales: Dict[str, int] = rospy.get_param('~axis_force_scales')
        self._axis_setpoint_scales: Dict[str, int] = rospy.get_param('~axis_setpoint_scales')


def main():
    rospy.init_node('teleop_planner')
    p = TeleopPlanner()
    p.start()