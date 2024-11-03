import rospy
import numpy as np
import tf
from geometry_msgs.msg import Vector3Stamped
from dataclasses import dataclass

from tauv_msgs.msg import PingDetection
from tauv_util import tm, tl

@dataclass
class Line:
    point: np.ndarray
    unit: np.ndarray

class PingerVoter:
    def __init__(self, pinger_A_pos, pinger_B_pos, freq, freq_thresh=100, voting_dist_thresh=10, consensus_thresh=5) -> None:
        self._pinger_A_pos = pinger_A_pos
        self._pinger_B_pos = pinger_B_pos
        self._freq = freq
        self._freq_thresh = freq_thresh
        self._voting_dist_thresh = voting_dist_thresh
        self._consensus_thresh = consensus_thresh

        self.pinger_A_votes = 0
        self.pinger_B_votes = 0

    def _vector_similarity(self, v1, v2):
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        cos_sim = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)

        angular_similarity = 1 - np.arccos(cos_sim) / np.pi

        return angular_similarity
    
    def _point_to_line_dist(self, q, l: Line):
        top = np.linalg.norm(np.cross(q - l.point, l.unit))
        bot = np.linalg.norm(l.unit)

        return top / bot

    def update(self, msg: PingDetection):
        direction_time = rospy.Time(
            secs=msg.header.stamp.secs,
            nsecs=msg.header.stamp.nsecs
        )

        try:
            direction_stamped = Vector3Stamped(
                header=msg.header,
                vector=msg.direction
            )

            world_direction = tf.TransformerROS.transformVector3(target_frame='kf/odom', v3s=direction_stamped)
            world_direction_vector = np.array(tl(world_direction.vector))
            world_direction_vector /= np.linalg.norm(world_direction_vector)

            vehicle_pos, _ = tf.Transformer.lookupTransform(target_frame='kf/odom', source_frame=msg.header.frame_id, time=direction_time)
            vehicle_pos = np.array(vehicle_pos)
        except Exception as e:
            rospy.logwarn(f"Could not look up vehicle transform for converting direction vector: {e}")
            return

        pinger_line = Line(
            point=vehicle_pos,
            unit=world_direction_vector
        )

        if self._freq - self._freq_thresh <= msg.frequency <= self._freq + self._freq_thresh:
            direction_pinger_A_dist = self._point_to_line_dist(self._pinger_A_pos, pinger_line)
            direction_pinger_B_dist = self._point_to_line_dist(self._pinger_B_pos, pinger_line)

            if max(direction_pinger_A_dist, direction_pinger_B_dist) <= self._voting_dist_thresh:
                if direction_pinger_A_dist < direction_pinger_B_dist:
                    self.pinger_A_votes += 1
                else:
                    self.pinger_B_votes += 1
        # How to decide which one to use?
        
