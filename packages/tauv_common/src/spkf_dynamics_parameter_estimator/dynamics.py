import numpy as np
from math import cos, sin

# params = {
# m, v,
# gx, gy, gz,
# bx, by, bz,
# Ixx, Ixy, Ixz, Iyy, Iyz, Izz,
# dlu, dlv, dlw, dlp, dlq, dlr,
# dqu, dqv, dqw, dqp, dqq, dqr,
# amu, amv, amw, amp, amq, amr
# }

# state = {
# phi, theta, psi,
# u, v, w,
# p, q, r
# }

# wrench = {
# fx, fy, fz
# tx, ty, tz
# }

def get_parameter_names() -> [str]:
    return [
        'm', 'v',
        'gx', 'gy', 'gz',
        'bx', 'by', 'bz',
        'Ixx', 'Ixy', 'Ixz', 'Iyy', 'Iyz', 'Izz',
        'dlu', 'dlv', 'dlw', 'dlp', 'dlq', 'dlr',
        'dqu', 'dqv', 'dqw', 'dqp', 'dqq', 'dqr',
        'amu', 'amv', 'amw', 'amp', 'amq', 'amr'
    ]

def get_acceleration(params: np.array, state: np.array, wrench: np.array) -> np.array:
    assert params.shape == (32,)
    assert state.shape == (9,)
    assert wrench.shape == (6,)

    M = get_M_body(params) + get_M_added(params)
    C = get_C_body(params, state) + get_C_added(params, state)
    D = get_D(params, state)
    G = get_G(params, state)

    v = state[3:9]

    acceleration = np.linalg.inv(M) @ (wrench - C @ v - D @ v - G)
    return acceleration

def get_R_world_body(state: np.array) -> np.array:
    assert state.shape == (9,)

    cphi = cos(state[0])
    sphi = sin(state[0])
    cth = cos(state[1])
    sth = sin(state[1])
    cpsi = cos(state[2])
    spsi = sin(state[2])

    return np.array([
        [cpsi * cth, -spsi * cphi + cpsi * sth * sphi, spsi * sphi + cpsi * cphi * sth],
        [spsi * cth, cpsi * cphi + sphi * sth * spsi, -cpsi * sphi + sth * spsi * cth],
        [-sth, cth * sphi, cth * cphi]
    ])

def get_M_body(params: np.array) -> np.array:
    assert params.shape == (32,)

    m = params[0]
    (gx, gy, gz) = params[2:5]
    (Ixx, Ixy, Ixz, Iyy, Iyz, Izz) = params[8:14]

    return np.array([
        [m, 0, 0, 0, m * gz, -m * gy],
        [0, m, 0, -m * gz, 0, m * gx],
        [0, 0, m, m * gy, -m * gx, 0],
        [0, -m * gz, m * gy, Ixx, -Ixy, -Ixz],
        [m * gz, 0, -m * gx, -Ixy, Iyy, -Iyz],
        [-m * gy, m * gx, 0, -Ixz, -Iyz, Izz]
    ])

def get_C_body(params: np.array, state: np.array) -> np.array:
    assert params.shape == (32,)
    assert state.shape == (9,)

    m = params[0]
    (gx, gy, gz) = params[2:5]
    (Ixx, Ixy, Ixz, Iyy, Iyz, Izz) = params[8:14]
    (u, v, w, p, q, r) = state[3:9]

    C_12 = np.array([
        [m * (gy * q + gz * r), -m * (gx * q - w), -m * (gx * r + v)],
        [-m * (gy * p + w), m * (gz * r + gx * p), -m * (gy * r - u)],
        [-m * (gz * p - v), -m * (gz * q + u), m * (gx * p + gy * q)],
    ])
    C_22 = np.array([
        [0, -Iyz * q - Ixz * p + Izz * r, Iyz * r + Ixy * p - Iyy * q],
        [Ixy * q + Ixz * p - Izz * r, 0, Ixz * r + Ixy * q + Ixx * p],
        [-Iyz * r - Ixy * p - Iyy * q, Ixz * r + Ixy * q - Ixx * p, 0]
    ])
    return np.vstack((
        np.hstack((np.zeros((3, 3)), C_12)),
        np.hstack((-C_12.T, C_22))
    ))

def get_M_added(params: np.array) -> np.array:
    assert params.shape == (32,)

    (au, av, aw, ap, aq, ar) = params[26:32]

    return np.diag([au, av, aw, ap, aq, ar])

def get_C_added(params: np.array, state: np.array) -> np.array:
    assert params.shape == (32,)
    assert state.shape == (9,)

    (au, av, aw, ap, aq, ar) = params[26:32]
    (u, v, w, p, q, r) = state[3:9]

    return np.array([
        [0, 0, 0, 0, aw * w, -av * v],
        [0, 0, 0, -aw * w, 0, au * u],
        [0, 0, 0, av * v, -au * u, 0],
        [0, aw * w, -av * v, 0, ar * r, -aq * q],
        [-aw * w, 0, au * u, -ar * r, 0, ap * p],
        [av * v, -au * u, 0, aq * q, -ap * p, 0]
    ])

def get_D(params: np.array, state: np.array) -> np.array:
    assert params.shape == (32,)
    assert state.shape == (9,)

    d = params[14:20]
    dd = params[20:26]
    v = state[3:9]

    return np.diag(d + dd * v)

def get_G(params: np.array, state: np.array) -> np.array:
    assert params.shape == (32,)
    assert state.shape == (9,)

    m = params[0]
    v = params[1]
    (gx, gy, gz) = params[2:5]
    (bx, by, bz) = params[5:8]
    (phi, theta, psi) = state[0:3]

    W = 9.81 * m
    B = 9.81 * v * 1028.0

    return np.array([
        (W - B) * sin(theta),
        -(W - B) * cos(theta) * sin(phi),
        -(W - B) * cos(theta) * cos(phi),
        -(gy * W - by * B) * cos(theta) * cos(phi) + (gz * W - bz * B) * cos(theta) * sin(phi),
        (gz * W - bz * B) * sin(theta) + (gx * W - bx * B) * cos(theta) * cos(phi),
        -(gx * W - bx * B) * cos(theta) * sin(phi) - (gy * W - by * B) * sin(theta)
    ])


def S(v: np.array) -> np.array:
    assert v.shape == (3,)

    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
