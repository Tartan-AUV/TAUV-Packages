import rospy
import numpy as np
from typing import Optional
from threading import Lock

from geometry_msgs.msg import WrenchStamped
from tauv_msgs.msg import NavigationState, DynamicsParametersEstimate
from tauv_msgs.srv import UpdateDynamicsParameterConfigs, UpdateDynamicsParameterConfigsRequest, UpdateDynamicsParameterConfigsResponse
from tauv_util.transforms import euler_velocity_to_axis_velocity, euler_acceleration_to_axis_acceleration
from tauv_util.types import vector_to_numpy

from . import dynamics
from .parameter_spkf import ParameterSPKF

# Want to be able to fix some parameters and estimate others while the parameter estimator is running
# SPKF will be reinitialized each time

# Fix a parameter to a given value
# Set process covariance for a parameter to a given value
# Set measurement covariance for acceleration to a value

class ParameterConfig:

    def __init__(self, initial_value: float, truth_value: float, fixed: bool, initial_covariance: float, process_covariance: float, limits: (float, float)):
        self.initial_value = initial_value
        self.truth_value = truth_value
        self.fixed = fixed
        self.initial_covariance = initial_covariance
        self.process_covariance = process_covariance
        self.limits = limits

    def __str__(self) -> str:
        return f'initial_value: {self.initial_value}, truth_value: {self.truth_value}, fixed: {self.fixed}, initial_covariance: {self.initial_covariance}, process_covariance: {self.process_covariance}, limits: {self.limits}'

class DynamicsParameterEstimator:

    def __init__(self):
        self._load_config()

        self._dt: float = 1.0 / self._frequency

        self._wrench: Optional[WrenchStamped] = None
        self._nav_state: Optional[NavigationState] = None

        self._build_spkf()

        self._wrench_sub: rospy.Subscriber = rospy.Subscriber(
            'gnc/target_wrench', WrenchStamped, self._handle_wrench
        )
        self._nav_state_sub: rospy.Subscriber = rospy.Subscriber(
            'gnc/navigation_state', NavigationState, self._handle_nav_state
        )
        self._parameters_pub: rospy.Publisher = rospy.Publisher(
           'gnc/dynamics_parameters', DynamicsParametersEstimate, queue_size=10
        )
        self._truth_pub: rospy.Publisher = rospy.Publisher(
            'gnc/dynamics_parameters_truth', DynamicsParametersEstimate, queue_size=10
        )
        self._config_srv: rospy.Service = rospy.Service(
            'gnc/update_dynamics_parameter_configs', UpdateDynamicsParameterConfigs, self._handle_update_configs
        )

        self._config_lock = Lock()

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _update(self, timer_event):
        if self._wrench is None or self._nav_state is None:
            return

        self._config_lock.acquire()

        orientation = vector_to_numpy(self._nav_state.orientation)
        euler_velocity = vector_to_numpy(self._nav_state.euler_velocity)
        euler_acceleration = vector_to_numpy(self._nav_state.euler_acceleration)

        axis_velocity = euler_velocity_to_axis_velocity(orientation, euler_velocity)
        axis_acceleration = euler_acceleration_to_axis_acceleration(orientation, euler_velocity, euler_acceleration)

        state = np.concatenate((
            orientation,
            vector_to_numpy(self._nav_state.linear_velocity),
            axis_velocity
        ))

        wrench = np.concatenate((
            vector_to_numpy(self._wrench.wrench.force),
            vector_to_numpy(self._wrench.wrench.torque),
        ))

        acceleration = np.concatenate((
            vector_to_numpy(self._nav_state.linear_acceleration),
            axis_acceleration
        ))

        try:
            self._spkf.update(state=state, input=wrench, measurement=acceleration)
        except np.linalg.LinAlgError as e:
            rospy.logerr_throttle(1.0, f'Update linear algebra error: {e}')

        free_params = self._spkf.get_parameters()
        free_cov = np.diag(self._spkf.get_parameter_covariance())

        params, cov = self._merge(free_params, free_cov)

        params_msg = DynamicsParametersEstimate()
        params_msg.stamp = rospy.Time.now()
        params_msg.m = params[0]
        params_msg.v = params[1]
        params_msg.g = params[2:5].tolist()
        params_msg.b = params[5:8].tolist()
        params_msg.I = params[8:14].tolist()
        params_msg.dl = params[14:20].tolist()
        params_msg.dq = params[20:26].tolist()
        params_msg.am = params[26:32].tolist()
        params_msg.cov_m = cov[0]
        params_msg.cov_v = cov[1]
        params_msg.cov_g = cov[2:5].tolist()
        params_msg.cov_b = cov[5:8].tolist()
        params_msg.cov_I = cov[8:14].tolist()
        params_msg.cov_dl = cov[14:20].tolist()
        params_msg.cov_dq = cov[20:26].tolist()
        params_msg.cov_am = cov[26:32].tolist()
        self._parameters_pub.publish(params_msg)

        truth_msg = DynamicsParametersEstimate()
        truth_msg.stamp = rospy.Time.now()
        truth_msg.m = self._param_configs['m'].truth_value
        truth_msg.v = self._param_configs['v'].truth_value
        truth_msg.g[0] = self._param_configs['gx'].truth_value
        truth_msg.g[1] = self._param_configs['gy'].truth_value
        truth_msg.g[2] = self._param_configs['gz'].truth_value
        truth_msg.b[0] = self._param_configs['bx'].truth_value
        truth_msg.b[0] = self._param_configs['by'].truth_value
        truth_msg.b[0] = self._param_configs['bz'].truth_value
        truth_msg.I[0] = self._param_configs['Ixx'].truth_value
        truth_msg.I[1] = self._param_configs['Ixy'].truth_value
        truth_msg.I[2] = self._param_configs['Ixz'].truth_value
        truth_msg.I[3] = self._param_configs['Iyy'].truth_value
        truth_msg.I[4] = self._param_configs['Iyz'].truth_value
        truth_msg.I[5] = self._param_configs['Izz'].truth_value
        truth_msg.dl[0] = self._param_configs['dlu'].truth_value
        truth_msg.dl[1] = self._param_configs['dlv'].truth_value
        truth_msg.dl[2] = self._param_configs['dlw'].truth_value
        truth_msg.dl[3] = self._param_configs['dlp'].truth_value
        truth_msg.dl[4] = self._param_configs['dlq'].truth_value
        truth_msg.dl[5] = self._param_configs['dlr'].truth_value
        truth_msg.dq[0] = self._param_configs['dqu'].truth_value
        truth_msg.dq[1] = self._param_configs['dqv'].truth_value
        truth_msg.dq[2] = self._param_configs['dqw'].truth_value
        truth_msg.dq[3] = self._param_configs['dqp'].truth_value
        truth_msg.dq[4] = self._param_configs['dqq'].truth_value
        truth_msg.dq[5] = self._param_configs['dqr'].truth_value
        truth_msg.am[0] = self._param_configs['amu'].truth_value
        truth_msg.am[1] = self._param_configs['amv'].truth_value
        truth_msg.am[2] = self._param_configs['amw'].truth_value
        truth_msg.am[3] = self._param_configs['amp'].truth_value
        truth_msg.am[4] = self._param_configs['amq'].truth_value
        truth_msg.am[5] = self._param_configs['amr'].truth_value
        self._truth_pub.publish(truth_msg)

        self._config_lock.release()

    def _handle_wrench(self, msg: WrenchStamped):
        self._wrench = msg

    def _handle_nav_state(self, msg: NavigationState):
        self._nav_state = msg

    def _handle_update_configs(self, req: UpdateDynamicsParameterConfigsRequest) -> UpdateDynamicsParameterConfigsResponse:
        self._config_lock.acquire()
        res = UpdateDynamicsParameterConfigsResponse()
        res.success = True

        reset = {name: False for name in self._param_names}

        old_free_params = self._spkf.get_parameters()
        old_free_cov = np.diag(self._spkf.get_parameter_covariance())
        old_params, old_cov = self._merge(old_free_params, old_free_cov)

        for update in req.updates:
            if update.name not in self._param_names:
                res.success = False
                rospy.logwarn(f'Attempted to update configuration for {update.name}, ignoring')
                continue

            config = self._param_configs[update.name]

            if update.update_initial_value:
                config.initial_value = update.initial_value

            if update.update_fixed:
                config.fixed = update.fixed

            if update.update_initial_covariance:
                config.initial_covariance = update.initial_covariance

            if update.update_process_covariance:
                config.process_covariance = update.process_covariance

            if update.update_limits:
                config.limits = tuple(update.limits)

            if update.reset:
                reset[update.name] = True

            rospy.loginfo(f'Updated configuration for {update.name}: {config}')

        self._build_spkf()

        new_params = np.zeros(len(self._param_names))
        new_cov = np.zeros(len(self._param_names))

        for i, name in enumerate(self._param_names):
            if reset[name]:
                new_params[i] = self._param_configs[name].initial_value
                new_cov[i] = self._param_configs[name].initial_covariance
            else:
                new_params[i] = old_params[i]
                new_cov[i] = old_cov[i]

            # Switching from fixed: false to fixed: true resets to initial value
            # Doing otherwise would require setting the initial value in the config to the previous value

        new_free_params, new_free_cov = self._unmerge(new_params, new_cov)

        self._spkf.set_parameters(new_free_params)
        self._spkf.set_parameter_covariance(np.diag(new_free_cov))

        self._config_lock.release()
        return res

    def _load_config(self):
        self._frequency: int = rospy.get_param('~frequency')
        self._error_covariance: np.array = np.diag(rospy.get_param('~error_covariance'))

        self._param_names = dynamics.get_parameter_names()
        self._param_configs: {str: ParameterConfig} = {}

        configs = rospy.get_param('~parameters')

        for name in self._param_names:
            config = configs[name]

            self._param_configs[name] = ParameterConfig(
                initial_value=float(config.get('initial_value')),
                truth_value=float(config.get('truth_value')),
                fixed=bool(config.get('fixed')),
                initial_covariance=float(config.get('initial_covariance')),
                process_covariance=float(config.get('process_covariance')),
                limits=(float(config.get('limits')[0]), float(config.get('limits')[1]))
            )

    def _build_spkf(self):
        free_parameter_names = self._get_free_param_names()

        def measurement_function(state: np.array, input: np.array, free_parameters: np.array, error: np.array) -> np.array:
            parameters, _ = self._merge(free_parameters, None)

            return dynamics.get_acceleration(parameters, state, input) + error

        dim_parameters = len(free_parameter_names)
        initial_parameters = np.array([self._param_configs[name].initial_value for name in free_parameter_names])
        initial_covariance = np.diag([self._param_configs[name].initial_covariance for name in free_parameter_names])
        process_covariance = np.diag([self._param_configs[name].process_covariance for name in free_parameter_names])
        lower_parameter_limits = np.array([self._param_configs[name].limits[0] for name in free_parameter_names])
        upper_parameter_limits = np.array([self._param_configs[name].limits[1] for name in free_parameter_names])
        parameter_limits = np.column_stack((
            lower_parameter_limits,
            upper_parameter_limits
        ))

        self._spkf = ParameterSPKF(
            measurement_function=measurement_function,
            initial_parameters=initial_parameters,
            initial_covariance=initial_covariance,
            process_covariance=process_covariance,
            error_covariance=self._error_covariance,
            dim_parameters=dim_parameters,
            dim_state=9,
            dim_input=6,
            dim_measurement=6,
            parameter_limits=parameter_limits,
        )

    def _merge(self, free_params: np.array, free_cov: Optional[np.array]) -> (np.array, Optional[np.array]):
        free_param_names = self._get_free_param_names()

        params = np.zeros(len(self._param_names))
        cov = np.zeros(len(self._param_names))

        for i, name in enumerate(self._param_names):
            if self._param_configs[name].fixed:
                params[i] = self._param_configs[name].initial_value

                if free_cov is not None:
                    cov[i] = self._param_configs[name].initial_covariance
            else:
                free_i = free_param_names.index(name)
                params[i] = free_params[free_i]

                if free_cov is not None:
                    cov[i] = free_cov[free_i]

        if free_cov is not None:
            return params, cov
        else:
            return params, None

    def _unmerge(self, params: np.array, cov: Optional[np.array]) -> (np.array, Optional[np.array]):
        free_param_names = self._get_free_param_names()

        free_params = np.zeros(len(free_param_names))
        free_cov = np.zeros(len(free_param_names))

        for free_i, name in enumerate(free_param_names):
            i = self._param_names.index(name)
            free_params[free_i] = params[i]

            if cov is not None:
                free_cov[free_i] = cov[i]

        if free_cov is not None:
            return free_params, free_cov
        else:
            return free_params, None

    def _get_free_param_names(self) -> [str]:
        return [name for name in self._param_names if not self._param_configs[name].fixed]

def main():
    rospy.init_node('dynamics_parameter_estimator')
    d = DynamicsParameterEstimator()
    d.start()