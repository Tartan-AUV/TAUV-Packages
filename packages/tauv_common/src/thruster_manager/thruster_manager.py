import rospy
import numpy as np
from typing import Optional
import tf2_ros as tf2
from tauv_util.transforms import tf2_transform_to_translation, tf2_transform_to_quat, quat_to_rotm
from tauv_util.types import tl

from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float64

class ThrusterManager:

    def __init__(self):
        self._load_config()

        self._dt: float = 1.0 / self._frequency
        self._num_thrusters: int = len(self._thruster_ids)
        self._wrench: Optional[WrenchStamped] = None

        self._tf_buffer: tf2.Buffer = tf2.Buffer()
        self._tf_listener: tf2.TransformListener = tf2.TransformListener(self._tf_buffer)

        self._wrench_sub: rospy.Subscriber = rospy.Subscriber('gnc/target_wrench', WrenchStamped, self._handle_wrench)
        self._target_thrust_pubs: [rospy.Publisher] = []

        for thruster_id in self._thruster_ids:
            target_thrust_pub = rospy.Publisher(
                f'vehicle/thrusters/{thruster_id}/target_thrust',
                Float64,
                queue_size=10
            )
            self._target_thrust_pubs.append(target_thrust_pub)

        self._build_tam()

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _command_thrusts(self, thrusts: [float]):
        for (i, thrust) in enumerate(thrusts):
            thrust_msg = Float64()
            thrust_msg.data = thrust
            self._target_thrust_pubs[i].publish(thrust_msg)

    def _update(self, timer_event):
        if self._wrench is None:
            return

        tau = np.hstack((
            tl(self._wrench.wrench.force),
            tl(self._wrench.wrench.torque)
        ))

        thrusts = self._inv_tam @ tau

        self._command_thrusts(thrusts)

    def _handle_wrench(self, wrench: WrenchStamped):
        self._wrench = wrench

    def _build_tam(self):
        tam: np.array = np.zeros((6, self._num_thrusters))

        for (i, thruster_id) in enumerate(self._thruster_ids):
            base_frame = f'{self._tf_namespace}/vehicle'
            thruster_frame = f'{self._tf_namespace}/thruster_{thruster_id}'
            current_time = rospy.Time.now()
            try:
                transform = self._tf_buffer.lookup_transform(
                    base_frame,
                    thruster_frame,
                    current_time,
                    rospy.Duration(30.0)
                )

                trans = tf2_transform_to_translation(transform)
                quat = tf2_transform_to_quat(transform)

                print(f'transform from {base_frame} to {thruster_frame}')
                print(trans)
                print(quat)

                rotm = quat_to_rotm(quat)

                force = rotm @ np.array([1, 0, 0])
                torque = np.cross(trans, force)

                tau = np.hstack((force, torque)).transpose()

                tam[:, i] = tau

            except (tf2.LookupException, tf2.ConnectivityException, tf2.ExtrapolationException) as e:
                rospy.logerr(f'Could not get transform from {base_frame} to {thruster_frame}: {e}')

        print('tam:', tam)
        self._inv_tam = np.linalg.pinv(tam)
        print('inv_tam:', self._inv_tam)

    def _load_config(self):
        self._tf_namespace = rospy.get_param('tf_namespace')
        self._frequency = rospy.get_param('~frequency')
        self._thruster_ids = np.array(rospy.get_param('~thruster_ids'))

def main():
    rospy.init_node('thruster_manager')
    t = ThrusterManager()
    t.start()