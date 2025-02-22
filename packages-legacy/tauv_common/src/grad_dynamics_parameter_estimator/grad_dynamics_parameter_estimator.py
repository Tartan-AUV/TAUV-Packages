import rospy
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Dict, Optional, List
from dataclasses import dataclass
from threading import Lock

from geometry_msgs.msg import WrenchStamped
from tauv_msgs.msg import NavigationState, DynamicsParametersEstimate, DynamicsTuning
from tauv_msgs.srv import TuneDynamics, TuneDynamicsRequest
from tauv_util.spatialmath import ros_vector3_to_r3
from tauv_util.transforms import euler_velocity_to_axis_velocity, euler_acceleration_to_axis_acceleration

from .model import Param, param_labels, get_acceleration


@dataclass
class ParameterConfig:
    initial: float
    fixed: bool
    min: float
    max: float
    lr: float


class GradDynamicsParameterEstimator:

    def __init__(self):
        self._load_config()

        self._wrench: Optional[WrenchStamped] = None
        self._nav_state: Optional[NavigationState] = None

        self._batch_wrench: List[torch.Tensor] = []
        self._batch_accel: List[torch.Tensor] = []
        self._batch_state: List[torch.Tensor] = []

        self._wrench_sub: rospy.Subscriber = rospy.Subscriber(
            'gnc/target_wrench', WrenchStamped, self._handle_wrench
        )
        self._nav_state_sub: rospy.Subscriber = rospy.Subscriber(
            'gnc/navigation_state', NavigationState, self._handle_nav_state
        )
        self._parameters_pub: rospy.Publisher = rospy.Publisher(
            'gnc/dynamics_parameters', DynamicsParametersEstimate, queue_size=10
        )
        self._tune_dynamics_srv: rospy.ServiceProxy = rospy.ServiceProxy('gnc/tune_dynamics', TuneDynamics)
        self._init()

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _init(self):
        param_labels_inv: Dict[str, Param] = {
            v: k for (k, v) in param_labels.items()
        }

        self._param: torch.Tensor = torch.zeros((32,))
        self._param_min: torch.Tensor = torch.zeros((32,))
        self._param_max: torch.Tensor = torch.zeros((32,))
        self._param_fixed: torch.Tensor = torch.ones((32,), dtype=torch.bool)
        self._param_lr: torch.Tensor = torch.ones((32,))

        for (label, config) in self._param_configs.items():
            param = param_labels_inv[label]
            self._param[param] = config.initial
            self._param_min[param] = config.min
            self._param_max[param] = config.max
            self._param_fixed[param] = config.fixed
            self._param_lr[param] = config.lr

    def _update(self, timer_event):
        if self._wrench is None or self._nav_state is None:
            return

        position = ros_vector3_to_r3(self._nav_state.position)

        if position[2] < self._min_z:
            return

        orientation = ros_vector3_to_r3(self._nav_state.orientation)
        linear_velocity = ros_vector3_to_r3(self._nav_state.linear_velocity)
        linear_acceleration = ros_vector3_to_r3(self._nav_state.linear_acceleration)
        euler_velocity = ros_vector3_to_r3(self._nav_state.euler_velocity)
        euler_acceleration = ros_vector3_to_r3(self._nav_state.euler_acceleration)

        axis_velocity = euler_velocity_to_axis_velocity(orientation, euler_velocity)
        axis_acceleration = euler_acceleration_to_axis_acceleration(orientation, euler_velocity, euler_acceleration)

        force = ros_vector3_to_r3(self._wrench.wrench.force)
        torque = ros_vector3_to_r3(self._wrench.wrench.torque)

        # Add this as a sample to the batch
        state = torch.cat((
            torch.Tensor(position),
            torch.Tensor(orientation),
            torch.Tensor(linear_velocity),
            torch.Tensor(axis_velocity),
        ), axis=0)
        accel = torch.cat((
            torch.Tensor(linear_acceleration),
            torch.Tensor(axis_acceleration),
        ), axis=0)
        wrench = torch.cat((
            torch.Tensor(force),
            torch.Tensor(torque),
        ), axis=0)

        self._batch_state.append(state)
        self._batch_accel.append(accel)
        self._batch_wrench.append(wrench)

        # Then check if the batch is full
        if len(self._batch_state) < self._batch_size:
            return

        # Now actually do the batch update

        batch_state = torch.stack(self._batch_state, axis=0)
        batch_accel = torch.stack(self._batch_accel, axis=0)
        batch_wrench = torch.stack(self._batch_wrench, axis=0)
        self._batch_state = []
        self._batch_accel = []
        self._batch_wrench = []

        model_param = Variable(self._param.clone(), requires_grad=True)

        est_accel = get_acceleration(model_param.unsqueeze(0).repeat(self._batch_size, 1), batch_state, batch_wrench)

        print((est_accel - batch_accel).abs().mean())

        loss = F.mse_loss(est_accel, batch_accel, reduction="mean")
        loss.backward()

        self._param.data = torch.where(
            self._param_fixed,
            self._param.data,
            torch.clip(self._param.data - self._param_lr * model_param.grad.data, self._param_min, self._param_max)
        )
        model_param.grad.data.zero_()

        # print(self._param)
        # print(self._param_min)
        # print(self._param_max)
        print(est_accel)
        print(batch_accel)
        print(self._param_lr)

        self._publish_params()

        if self._should_update_controller:
            self._update_controller()

    def _handle_wrench(self, msg: WrenchStamped):
        self._wrench = msg

    def _handle_nav_state(self, msg: NavigationState):
        self._nav_state = msg

    def _publish_params(self):
        params_msg = DynamicsParametersEstimate()
        params_msg.stamp = rospy.Time.now()
        params_msg.m = float(self._param[Param.m])
        params_msg.v = float(self._param[Param.v])
        params_msg.g = self._param[Param.gx:Param.gz+1].tolist()
        params_msg.b = self._param[Param.bx:Param.bz+1].tolist()
        params_msg.I = self._param[Param.Ixx:Param.Izz+1].tolist()
        params_msg.dl = self._param[Param.dlu:Param.dlr+1].tolist()
        params_msg.dq = self._param[Param.dqu:Param.dqr+1].tolist()
        params_msg.am = self._param[Param.amu:Param.amr+1].tolist()
        self._parameters_pub.publish(params_msg)

    def _update_controller(self):
        t = DynamicsTuning()

        t.update_mass = True
        t.mass = float(self._param[Param.m])

        t.update_volume = True
        t.volume = float(self._param[Param.v])

        t.update_water_density = True
        t.water_density = 1028.0

        t.update_center_of_gravity = True
        t.center_of_gravity = self._param[Param.gx:Param.gz+1].tolist()

        t.update_center_of_buoyancy = True
        t.center_of_buoyancy = self._param[Param.bx:Param.bz+1].tolist()

        t.update_moments = True
        # t.moments = self._param[(Param.Ixx, Param.Iyy, Param.Izz, Param.Ixy, Param.Ixz, Param.Iyz)].tolist()
        # t.moments = self._param.gather(0, torch.Tensor([Param.Ixx, Param.Iyy, Param.Izz, Param.Ixy, Param.Ixz, Param.Iyz], dtype=torch.long)).tolist()
        t.moments = [
            float(self._param[Param.Ixx]),
            float(self._param[Param.Iyy]),
            float(self._param[Param.Izz]),
            float(self._param[Param.Ixy]),
            float(self._param[Param.Ixz]),
            float(self._param[Param.Iyz]),
        ]

        t.update_linear_damping = True
        t.linear_damping = self._param[Param.dlu:Param.dlr+1].tolist()

        t.update_quadratic_damping = True
        t.quadratic_damping = self._param[Param.dqu:Param.dqr+1].tolist()

        t.update_added_mass = True
        t.added_mass = self._param[Param.amu:Param.amr+1].tolist()

        req: TuneDynamicsRequest = TuneDynamicsRequest()
        req.tuning = t
        self._tune_dynamics_srv.call(req)

    def _load_config(self):
        self._frequency: float = rospy.get_param('~frequency')
        self._dt: float = 1.0 / self._frequency

        self._batch_size: int = int(rospy.get_param('~batch_size'))

        self._min_z: float = rospy.get_param('~min_z')

        self._param_configs: Dict[str, ParameterConfig] = {}

        self._should_update_controller: bool = bool(rospy.get_param('~update_controller'))

        configs = rospy.get_param('~parameters')

        for (name, config) in configs.items():
            self._param_configs[name] = ParameterConfig(
                initial=float(config['initial']),
                fixed=bool(config['fixed']),
                min=float(config['min']),
                max=float(config['max']),
                lr=float(config['lr']),
            )


def main():
    rospy.init_node('grad_dynamics_parameter_estimator')
    n = GradDynamicsParameterEstimator()
    n.start()
