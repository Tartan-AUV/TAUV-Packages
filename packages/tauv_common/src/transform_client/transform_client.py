import rospy
import tf2_ros
from typing import Optional
from spatialmath import SE3

from geometry_msgs.msg import TransformStamped

from tauv_util.spatialmath import ros_transform_to_se3, se3_to_ros_transform

class TransformClient:

    def __init__(self):
        self._tf_buffer: tf2_ros.Buffer = tf2_ros.Buffer()
        self._tf_listener: tf2_ros.TransformListener = tf2_ros.TransformListener(self._tf_buffer)
        self._tf_broadcaster: tf2_ros.TransformBroadcaster = tf2_ros.TransformBroadcaster()
        self._tf_static_broadcaster: tf2_ros.StaticTransformBroadcaster = tf2_ros.StaticTransformBroadcaster()

    def get_a_to_b(self, frame_a: str, frame_b: str,
                   time: rospy.Time = rospy.Time(0),
                   timeout: rospy.Duration = rospy.Duration(1)) -> SE3:
        tf_transform = self._tf_buffer.lookup_transform(frame_a, frame_b, time, timeout)
        return ros_transform_to_se3(tf_transform.transform)

    def set_a_to_b(self, frame_a: str, frame_b: str, tf_a_to_b: SE3, time: Optional[rospy.Time] = rospy.Time(0)):
        tf_transform = TransformStamped()
        tf_transform.header.frame_id = frame_a
        if time is not None:
            tf_transform.header.stamp = time
        tf_transform.child_frame_id = frame_b
        tf_transform.transform = se3_to_ros_transform(tf_a_to_b)

        if time is None:
            self._tf_static_broadcaster.sendTransform(tf_transform)
        else:
            self._tf_broadcaster.sendTransform(tf_transform)
