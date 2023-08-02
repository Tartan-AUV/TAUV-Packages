import rospy

from std_msgs.msg import Float64
import time


class ActuatorClient:

    def __init__(self):
        self._load_config()

        if self._has_torpedo:
            self._torpedo_pub: rospy.Publisher = rospy.Publisher(f'vehicle/servos/{self._torpedo_servo}/target_position', Float64)

        if self._has_marker:
            self._marker_pub: rospy.Publisher = rospy.Publisher(f'vehicle/servos/{self._marker_servo}/target_position', Float64)

        if self._has_arm:
            self._arm_pub: rospy.Publisher = rospy.Publisher(f'vehicle/servos/{self._arm_servo}/target_position', Float64)

        if self._has_suction:
            self._suction_pub: rospy.Publisher = rospy.Publisher(f'vehicle/servos/{self._suction_servo}/target_position', Float64)

    def shoot_torpedo(self, torpedo: int):
        if not self._has_torpedo:
            raise ValueError("torpedo is not enabled")

        if torpedo not in [0, 1]:
            raise ValueError(f"torpedo must be one of [0, 1], got {torpedo}")

        rospy.loginfo(f"shooting torpedo {torpedo}...")

        if torpedo == 0:
            self._torpedo_pub.publish(1.0)
            time.sleep(1.0)
            self._torpedo_pub.publish(0.0)
        else:
            self._torpedo_pub.publish(-1.0)
            time.sleep(1.0)
            self._torpedo_pub.publish(0.0)

    def drop_marker(self, marker: int):
        if not self._has_marker:
            raise ValueError("marker is not enabled")

        if marker not in [0, 1]:
            raise ValueError(f"marker must be one of [0, 1], got {marker}")

        rospy.loginfo(f"dropping marker {marker}...")

        if marker == 0:
            self._marker_pub.publish(1.0)
            time.sleep(1.0)
            self._marker_pub.publish(0.0)
        else:
            self._marker_pub.publish(-1.0)
            time.sleep(1.0)
            self._marker_pub.publish(0.0)

    def move_arm(self, position: float):
        if not self._has_arm:
            raise ValueError("arm is not enabled")

        if not (0 <= position <= 1):
            raise ValueError(f"position must be in [0, 1], got {position}")

        rospy.loginfo(f"moving arm to {position}...")

        self._arm_pub.publish(-position)

    def activate_suction(self, strength: float):
        if not self._has_suction:
            raise ValueError("suction is not enabled")

        if not (0 <= strength <= 1):
            raise ValueError(f"strength must be in [0, 1], got {strength}")

        rospy.loginfo(f"setting suction to {strength}...")

        self._suction_pub.publish(strength)

    def _load_config(self):
        self._has_torpedo: bool = rospy.get_param('actuators/has_torpedo')
        if self._has_torpedo:
            self._torpedo_servo: int = rospy.get_param('actuators/torpedo_servo')

        self._has_marker: bool = rospy.get_param('actuators/has_marker')
        if self._has_marker:
            self._marker_servo: int = rospy.get_param('actuators/marker_servo')

        self._has_arm: bool = rospy.get_param('actuators/has_arm')
        if self._has_arm:
            self._arm_servo: int = rospy.get_param('actuators/arm_servo')

        self._has_suction: bool = rospy.get_param('actuators/has_suction')
        if self._has_suction:
            self._suction_servo: int = rospy.get_param('actuators/suction_servo')
