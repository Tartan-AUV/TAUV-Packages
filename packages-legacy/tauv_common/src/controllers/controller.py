import rospy
import numpy as np
from typing import Optional

from std_srvs.srv import SetBool, SetBoolResponse

from dynamics.dynamics import Dynamics
from geometry_msgs.msg import WrenchStamped, Vector3
from tauv_msgs.msg import ControllerCommand, NavigationState, ControllerDebug
from tauv_msgs.srv import TuneDynamics, TuneDynamicsRequest, TuneDynamicsResponse, TuneController, TuneControllerRequest, TuneControllerResponse
from tauv_util.types import tl, tm
from tauv_util.pid import PID, pi_clip
from tauv_util.transforms import euler_velocity_to_axis_velocity



from tauv_alarms import Alarm, AlarmClient

import sys
print("PATHS PATHS PATHS")
print("\n".join(sys.path))


class Controller:

    def __init__(self):
        self._ac = AlarmClient()

        self._load_config()

        self._dt: float = 1.0 / self._frequency
        self._navigation_state: Optional[NavigationState] = None
        self._controller_command: Optional[ControllerCommand] = None
        self._dyn: Dynamics = Dynamics(
            m=self._m,
            v=self._v,
            rho=self._rho,
            r_G=self._r_G,
            r_B=self._r_B,
            I=self._I,
            D=self._D,
            D2=self._D2,
            Ma=self._Ma,
        )

        self._in_barrel_roll = False
        self._total_barrel_roll_angle = 0
        self._last_roll = 0

        self._build_pids()

        self._navigation_state_sub: rospy.Subscriber = rospy.Subscriber('gnc/navigation_state', NavigationState, self._handle_navigation_state)
        self._controller_command_sub: rospy.Subscriber = rospy.Subscriber('gnc/controller_command', ControllerCommand, self._handle_controller_command)
        self._controller_debug_pub = rospy.Publisher('gnc/controller_debug', ControllerDebug, queue_size=10)
        self._wrench_pub: rospy.Publisher = rospy.Publisher('gnc/target_wrench', WrenchStamped, queue_size=10)
        self._tune_dynamics_srv: rospy.Service = rospy.Service('gnc/tune_dynamics', TuneDynamics, self._handle_tune_dynamics)
        self._tune_controller_srv: rospy.Service = rospy.Service('gnc/tune_controller', TuneController, self._handle_tune_controller)
        self._do_barrel_roll: rospy.Service = rospy.Service('gnc/do_barrel_roll', SetBool, self._handle_do_barrel_roll)

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _update(self, timer_event):
        if self._navigation_state is None or self._controller_command is None:
            return

        state = self._navigation_state
        cmd = self._controller_command

        position = tl(state.position)
        orientation = tl(state.orientation)
        euler_velocity = tl(state.euler_velocity)
        linear_velocity = tl(state.linear_velocity)

        roll = orientation[0]

        if self._in_barrel_roll:
            self._total_barrel_roll_angle -= abs((roll - self._last_roll + np.pi) % (2*np.pi) - np.pi)
            print(self._total_barrel_roll_angle)
            if abs(self._total_barrel_roll_angle) < 2 * np.pi:
                cmd.use_setpoint_roll = False
                cmd.use_f_roll = True
                cmd.f_roll = -20

            else:
                self._in_barrel_roll = False
                self._total_barrel_roll_angle = 0


        self._last_roll = orientation[0]

        eta = np.concatenate((
            position,
            orientation
        ))

        axis_velocity = euler_velocity_to_axis_velocity(orientation, euler_velocity)
        v = np.concatenate((
            linear_velocity,
            axis_velocity
        ))

        z_error = position[2] - cmd.setpoint_z
        roll_error = orientation[0] - cmd.setpoint_roll
        pitch_error = orientation[1] - cmd.setpoint_pitch

        roll_velocity = euler_velocity[0]
        pitch_velocity = euler_velocity[1]
        z_velocity = linear_velocity[2]

        z_effort = self._pids[0](z_error, z_velocity)
        roll_effort = self._pids[1](roll_error, roll_velocity)
        pitch_effort = self._pids[2](pitch_error, pitch_velocity)

        vd = np.array([
            cmd.a_x,
            cmd.a_y,
            cmd.a_z if not cmd.use_setpoint_z else z_effort,
            cmd.a_roll if not cmd.use_setpoint_roll else roll_effort,
            cmd.a_pitch if not cmd.use_setpoint_pitch else pitch_effort,
            cmd.a_yaw
        ])

        tau = self._dyn.compute_tau(eta, v, vd)

        if cmd.use_f_x:
            tau[0] = cmd.f_x
        if cmd.use_f_y:
            tau[1] = cmd.f_y
        if cmd.use_f_z:
            tau[2] = cmd.f_z
        if cmd.use_f_roll:
            tau[3] = cmd.f_roll
        if cmd.use_f_pitch:
            tau[4] = cmd.f_pitch
        if cmd.use_f_yaw:
            tau[5] = cmd.f_yaw

        # TODO: Fix max wrench clamping
        # while not np.allclose(np.minimum(np.abs(tau), self._max_wrench), np.abs(tau)):
        #     tau = 0.75 * tau
        #     # TODO FIX THIS
        #     # tau = self._dyn.compute_tau(eta, v, vd)
        tau = np.sign(tau) * np.minimum(np.abs(tau), self._max_wrench)

        # Need to LPF the wrench here with a certain time constant
        wrench: WrenchStamped = WrenchStamped()
        wrench.header.frame_id = f'{self._tf_namespace}/vehicle'
        wrench.header.stamp = rospy.Time.now()
        wrench.wrench.force = Vector3(tau[0], tau[1], tau[2])
        wrench.wrench.torque = Vector3(tau[3], tau[4], tau[5])
        self._wrench_pub.publish(wrench)

        controller_debug: ControllerDebug = ControllerDebug()
        controller_debug.z.tuning = self._pids[0].get_tuning()
        controller_debug.z.value = position[2]
        controller_debug.z.error = z_error
        controller_debug.z.setpoint = cmd.setpoint_z
        controller_debug.z.proportional = self._pids[0]._proportional
        controller_debug.z.integral = self._pids[0]._integral
        controller_debug.z.derivative = self._pids[0]._derivative
        controller_debug.z.effort = z_effort
        controller_debug.roll.tuning = self._pids[1].get_tuning()
        controller_debug.roll.value = orientation[0]
        controller_debug.roll.error = roll_error
        controller_debug.roll.setpoint = cmd.setpoint_roll
        controller_debug.roll.proportional = self._pids[1]._proportional
        controller_debug.roll.integral = self._pids[1]._integral
        controller_debug.roll.derivative = self._pids[1]._derivative
        controller_debug.roll.effort = roll_effort
        controller_debug.pitch.tuning = self._pids[2].get_tuning()
        controller_debug.pitch.value = orientation[1]
        controller_debug.pitch.setpoint = cmd.setpoint_pitch
        controller_debug.pitch.error = pitch_error
        controller_debug.pitch.proportional = self._pids[2]._proportional
        controller_debug.pitch.integral = self._pids[2]._integral
        controller_debug.pitch.derivative = self._pids[2]._derivative
        controller_debug.pitch.effort = pitch_effort
        self._controller_debug_pub.publish(controller_debug)

        self._ac.clear(Alarm.CONTROLLER_NOT_INITIALIZED)

    def _handle_navigation_state(self, msg: NavigationState):
        self._navigation_state = msg

    def _handle_controller_command(self, msg: ControllerCommand):
        self._controller_command = msg

    def _handle_do_barrel_roll(self, msg):
        self._in_barrel_roll = True
        response = "barrel rolling"
        return SetBoolResponse(success=True, message=response)

    def _handle_tune_dynamics(self, req: TuneDynamicsRequest) -> TuneDynamicsResponse:
        if req.tuning.update_mass:
            self._m = req.tuning.mass

        if req.tuning.update_volume:
            self._v = req.tuning.volume

        if req.tuning.update_water_density:
            self._rho = req.tuning.water_density

        if req.tuning.update_center_of_gravity:
            self._r_G = req.tuning.center_of_gravity

        if req.tuning.update_center_of_buoyancy:
            self._r_B = req.tuning.center_of_buoyancy

        if req.tuning.update_moments:
            self._I = req.tuning.moments

        if req.tuning.update_linear_damping:
            self._D = req.tuning.linear_damping

        if req.tuning.update_quadratic_damping:
            self._D2 = req.tuning.quadratic_damping

        if req.tuning.update_added_mass:
            self._Ma = req.tuning.added_mass

        self._dyn: Dynamics = Dynamics(
            m = self._m,
            v = self._v,
            rho = self._rho,
            r_G = self._r_G,
            r_B = self._r_B,
            I = self._I,
            D = self._D,
            D2 = self._D2,
            Ma = self._Ma,
        )
        return TuneDynamicsResponse(True)

    def _handle_tune_controller(self, req: TuneControllerRequest) -> TuneControllerResponse:
        fields = {"z": 0, "roll": 1, "pitch": 2}

        for tuning in req.tunings:
            field = fields.get(tuning.axis)

            if field is None:
                return TuneControllerResponse(False)

            self._kp[field] = float(tuning.kp)
            self._ki[field] = float(tuning.ki)
            self._kd[field] = float(tuning.kd)
            self._tau[field] = float(tuning.tau)
            self._limits[field] = np.array(tuning.limits, dtype=np.float64)
        self._build_pids()
        return TuneControllerResponse(True)

    def _build_pids(self):
        pids = []

        # z, roll, pitch
        for i in range(3):
            pid = PID(
                Kp=self._kp[i],
                Ki=self._ki[i],
                Kd=self._kd[i],
                error_map=pi_clip,
                output_limits=self._limits[i],
                proportional_on_measurement=False,
                sample_time=self._dt,
                d_alpha=self._dt / self._tau[i] if self._tau[i] > 0 else 1
            )
            pids.append(pid)

        self._pids = pids

    def _load_config(self):
        self._tf_namespace = rospy.get_param('tf_namespace')
        self._frequency = rospy.get_param('~frequency')
        self._kp = np.array(rospy.get_param('~kp'))
        self._ki = np.array(rospy.get_param('~ki'))
        self._kd = np.array(rospy.get_param('~kd'))
        self._tau = np.array(rospy.get_param('~tau'))
        self._limits = np.array(rospy.get_param('~limits'))

        self._max_wrench = np.array(rospy.get_param('~max_wrench'))

        self._m = rospy.get_param('~dynamics/mass')
        self._v = rospy.get_param('~dynamics/volume')
        self._rho = rospy.get_param('~dynamics/water_density')
        self._r_G = np.array(rospy.get_param('~dynamics/center_of_gravity'))
        self._r_B = np.array(rospy.get_param('~dynamics/center_of_buoyancy'))
        self._I = np.array(rospy.get_param('~dynamics/moments'))
        self._D = np.array(rospy.get_param('~dynamics/linear_damping'))
        self._D2 = np.array(rospy.get_param('~dynamics/quadratic_damping'))
        self._Ma = np.array(rospy.get_param('~dynamics/added_mass'))



def main():
    rospy.init_node('controller')
    c = Controller()
    c.start()
