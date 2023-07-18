import rospy
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from math import floor, ceil
from typing import List

from .maestro import Maestro
from tauv_util.types import tl
from tauv_msgs.msg import Battery as BatteryMsg, Servos as ServosMsg
from std_msgs.msg import Float64
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from geometry_msgs.msg import Wrench, WrenchStamped
from std_msgs.msg import Bool

from tauv_alarms import Alarm, AlarmClient
from tauv_common.src.tauv_util.math import quadratic_roots

class Thrusters:
    def __init__(self):
        self._ac: AlarmClient = AlarmClient()

        self._dt: float = 0.02
        self._timeout: float = 1.0

        self._load_config()

        while not self._try_init():
            rospy.sleep(0.5)
        rospy.loginfo('initialized maestro')

        self._is_armed: bool = False
        self._arm_service: rospy.Service = rospy.Service('vehicle/thrusters/arm', SetBool, self._handle_arm)

        self._target_thrust_subs = []
        for thruster_id in range(len(self._thruster_channels)):
            target_thrust_sub = rospy.Subscriber(f'vehicle/thrusters/{thruster_id}/target_thrust', Float64, self._handle_target_thrust, callback_args=thruster_id)
            self._target_thrust_subs.append(target_thrust_sub)

        self._target_position_subs = []
        for servo_id in range(len(self._servo_channels)):
            target_position_sub = rospy.Subscriber(f'vehicle/servos/{servo_id}/target_position', Float64, self._handle_target_position, callback_args=servo_id)
            self._target_position_subs.append(target_position_sub)

        self._battery_sub: rospy.Subscriber = rospy.Subscriber('vehicle/battery', BatteryMsg, self._handle_battery)
        self._active_pub: rospy.Publisher = rospy.Publisher('vehicle/thrusters/active', Bool, queue_size=10)

        self._battery_voltage: float = self._default_battery_voltage
        self._thrust_update_time: rospy.Time = rospy.Time.now()
        self._target_thrusts = [0.0] * len(self._thruster_channels)
        self._target_positions = [0.0] * len(self._servo_channels)

    def _try_init(self):
        try:
            self._maestro = Maestro(ttyStr=self._maestro_port)
            return True
        except Exception as e:
            print(e)
            return False

    def start(self):
        rospy.loginfo('start')
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _update(self, timer_event):
        self._ac.set(Alarm.SUB_DISARMED, value=not self._is_armed)
        self._active_pub.publish(self._is_armed)

        if (rospy.Time.now() - self._thrust_update_time).to_sec() > self._timeout \
                or not self._is_armed:
            self._target_thrusts = [0] * len(self._thruster_channels)
            # self._thrust_update_time = rospy.Time.now()

        self._maestro.clearErrors()

        for (thruster, thrust) in enumerate(self._target_thrusts):
            self._set_thrust(thruster, thrust)

        for (servo, position) in enumerate(self._target_positions):
            self._set_position(servo, position)

        self._ac.clear(Alarm.THRUSTERS_NOT_INITIALIZED)

    def _handle_arm(self, req: SetBoolRequest):
        rospy.loginfo('armed' if req.data else 'disarmed')
        self._is_armed = req.data
        return SetBoolResponse(True, '')

    def _handle_battery(self, msg: BatteryMsg):
        self._battery_voltage = msg.voltage

    def _handle_target_thrust(self, msg: Float64, thruster_id: int):
        if self._is_armed:
            self._target_thrusts[thruster_id] = msg.data
            self._thrust_update_time = rospy.Time.now()

    def _handle_target_position(self, msg: Float64, servo_id: int):
        if self._is_armed:
            self._target_positions[servo_id] = msg.data

    def _set_thrust(self, thruster: int, thrust: float):
        pwm_speed = self._get_pwm_speed(thruster, thrust)

        self._maestro.setTarget(pwm_speed * 4, self._thruster_channels[thruster])

    def _set_position(self, servo: int, position: float):
        position = clamp(position, -180, 180)
        pwm_speed = floor((position / 90) * 1000 + 1500)
        self._maestro.setTarget(pwm_speed * 4, self._servo_channels[servo])

    def _get_pwm_speed(self, thruster: int, thrust: float) -> int:
        def select_voltage_bounds(v):
            if v < 10 or v > 20:
                raise LookupError
            
            ub = ceil(v / 2) * 2
            return ub - 2, ub

        pwm_speed = 1500
        thrust = thrust * self._thrust_inversions[thruster]
        
        try:
            lower_v, upper_v = select_voltage_bounds(self._battery_voltage)
        except LookupError:
            rospy.logwarn(f"Requested thruster PWMs beyond voltage bounds (at {self._battery_voltage}V)")
            return pwm_speed

        if thrust < 0 and -self._negative_max_thrust < thrust < -self._negative_min_thrust:
            lower_quadratic = self._rev_thrust_coefficients[lower_v]
            upper_quadratic = self._rev_thrust_coefficients[upper_v]
            selector = min
        elif thrust > 0 and self._positive_min_thrust < thrust < self._positive_max_thrust:
            lower_quadratic = self._fwd_thrust_coefficients[lower_v]
            upper_quadratic = self._fwd_thrust_coefficients[upper_v]
            selector = max
        else:
            return pwm_speed

        scale = (self._battery_voltage - lower_v) / 2.0
        a, b, c = (1 - scale) * lower_quadratic + scale * upper_quadratic

        try: 
            target_pwm = floor(selector(quadratic_roots(a, b, c - thrust)))

            if self._minimum_pwm_speed < target_pwm < self._maximum_pwm_speed:
                pwm_speed = target_pwm
            
        except ArithmeticError:
            rospy.logwarn(f"Requested thruster PWMs with no solution (at {thrust} kg F)")
        
        return pwm_speed

    def _load_config(self):
        self._maestro_port: str = rospy.get_param('~maestro_port')
        self._thruster_channels: List[int] = rospy.get_param('~thruster_channels')
        self._servo_channels: List[int] = rospy.get_param('~servo_channels')
        self._default_battery_voltage: float = rospy.get_param('~default_battery_voltage')
        self._minimum_pwm_speed: float = rospy.get_param('~minimum_pwm_speed')
        self._maximum_pwm_speed: float = rospy.get_param('~maximum_pwm_speed')
        self._negative_min_thrust: float = rospy.get_param('~negative_min_thrust')
        self._negative_max_thrust: float = rospy.get_param('~negative_max_thrust')
        self._positive_min_thrust: float = rospy.get_param('~positive_min_thrust')
        self._positive_max_thrust: float = rospy.get_param('~positive_max_thrust')
        self._fwd_thrust_coefficients: dict[int, np.array] = {
            int(k): np.array(v) for (k, v) in rospy.get_param('~fwd_thrust_coefficients').items()
        }
        self._rev_thrust_coefficients: dict[int, np.array] = {
            int(k): np.array(v) for (k, v) in rospy.get_param('~rev_thrust_coefficients').items()
        }
        self._thrust_inversions: List[float] = rospy.get_param('~thrust_inversions')

def clamp(x, x_min, x_max):
    return min(max(x, x_min), x_max)

def main():
    rospy.init_node('thrusters')
    t = Thrusters()
    t.start()