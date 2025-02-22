import numpy as np
from geometry_msgs.msg import Vector3, Quaternion, Point


# to list
def tl(o):
    if isinstance(o, Vector3):
        return np.array([o.x, o.y, o.z])
    if isinstance(o, Point):
        return np.array([o.x, o.y, o.z])
    if isinstance(o, Quaternion):
        return np.array([o.x, o.y, o.z, o.w])
    if isinstance(o, list):
        return np.array(o)
    raise ValueError("Unsupported type for tl! Add it in tauv_util/types.py")


# to msg
def tm(l, t):
    if t == Point:
        return Point(l[0], l[1], l[2])
    if t == Vector3:
        return Vector3(l[0], l[1], l[2])
    if t == Quaternion:
        return Quaternion(l[0], l[1], l[2], l[3])
    raise ValueError("Unsupported type for tm! Add it in tauv_util/types.py")

def vector_to_numpy(v: Vector3) -> np.array:
    return np.array([v.x, v.y, v.z])

def numpy_to_vector(a: np.array) -> Vector3:
    assert(a.shape == (3,))
    return Vector3(a[0], a[1], a[2])