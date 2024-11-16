import torch
from typing import List
from enum import IntEnum
from math import cos, sin, pi


class State(IntEnum):
    x = 0
    y = 1
    z = 2
    phi = 3
    theta = 4
    psi = 5
    u = 6
    v = 7
    w = 8
    p = 9
    q = 10
    r = 11

state_labels = {
    State.x: "x",
    State.y: "y",
    State.z: "z",
    State.phi: "phi",
    State.theta: "theta",
    State.psi: "psi",
    State.u: "u",
    State.v: "v",
    State.w: "w",
    State.p: "p",
    State.q: "q",
    State.r: "r",
}


class Param(IntEnum):
    m = 0
    v = 1
    gx = 2
    gy = 3
    gz = 4
    bx = 5
    by = 6
    bz = 7
    Ixx = 8
    Ixy = 9
    Ixz = 10
    Iyy = 11
    Iyz = 12
    Izz = 13
    dlu = 14
    dlv = 15
    dlw = 16
    dlp = 17
    dlq = 18
    dlr = 19
    dqu = 20
    dqv = 21
    dqw = 22
    dqp = 23
    dqq = 24
    dqr = 25
    amu = 26
    amv = 27
    amw = 28
    amp = 29
    amq = 30
    amr = 31

param_labels = {
    Param.m: "m",
    Param.v: "v",
    Param.gx: "gx",
    Param.gy: "gy",
    Param.gz: "gz",
    Param.bx: "bx",
    Param.by: "by",
    Param.bz: "bz",
    Param.Ixx: "Ixx",
    Param.Ixy: "Ixy",
    Param.Ixz: "Ixz",
    Param.Iyy: "Iyy",
    Param.Iyz: "Iyz",
    Param.Izz: "Izz",
    Param.dlu: "dlu",
    Param.dlv: "dlv",
    Param.dlw: "dlw",
    Param.dlp: "dlp",
    Param.dlq: "dlq",
    Param.dlr: "dlr",
    Param.dqu: "dqu",
    Param.dqv: "dqv",
    Param.dqw: "dqw",
    Param.dqp: "dqp",
    Param.dqq: "dqq",
    Param.dqr: "dqr",
    Param.amu: "amu",
    Param.amv: "amv",
    Param.amw: "amw",
    Param.amp: "amp",
    Param.amq: "amq",
    Param.amr: "amr",
}

g = 9.81       # acceleration due to gravity, m/s^2
rho = 1028.0   # density of water, kg / m^3

n_state = 12
n_params = 32

def get_acceleration(params: torch.Tensor, state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    M = get_M_body(params=params) + get_M_added(params=params)
    C = get_C_body(params=params, state=state) + get_C_added(params=params, state=state)
    D = get_D(params=params, state=state)
    G = get_G(params=params, state=state)

    v = state[:, (State.u, State.v, State.w, State.p, State.q, State.r)]

    a = torch.bmm(torch.linalg.inv(M), (tau - torch.bmm(C, v.unsqueeze(-1)).squeeze(-1) - torch.bmm(D, v.unsqueeze(-1)).squeeze(-1) - G).unsqueeze(-1)).squeeze(-1)

    return a


def get_wrench(params: torch.Tensor, state: torch.Tensor, vdot: torch.Tensor) -> torch.Tensor:
    M = get_M_body(params=params) + get_M_added(params=params)
    C = get_C_body(params=params, state=state) + get_C_added(params=params, state=state)
    D = get_D(params=params, state=state)
    G = get_G(params=params, state=state)

    v = state[:, (State.u, State.v, State.w, State.p, State.q, State.r)]

    tau = torch.bmm(M, vdot.unsqueeze(-1)).squeeze(-1) + torch.bmm(C, v.unsqueeze(-1)).squeeze(-1) + torch.bmm(D, v.unsqueeze(-1)).squeeze(-1) + G

    return tau


def get_M_body(params: torch.Tensor) -> torch.Tensor:
    m = params[:, Param.m.value]
    gx = params[:, Param.gx.value]
    gy = params[:, Param.gy.value]
    gz = params[:, Param.gz.value]
    Ixx = params[:, Param.Ixx.value]
    Ixy = params[:, Param.Ixy.value]
    Ixz = params[:, Param.Ixz.value]
    Iyy = params[:, Param.Iyy.value]
    Iyz = params[:, Param.Iyz.value]
    Izz = params[:, Param.Izz.value]

    zero = torch.zeros(params.shape[0])

    return torch.stack([
        torch.stack([m, zero, zero, zero, m * gz, -m * gy]),
        torch.stack([zero, m, zero, -m * gz, zero, m * gx]),
        torch.stack([zero, zero, m, m * gy, -m * gx, zero]),
        torch.stack([zero, -m * gz, m * gy, Ixx, -Ixy, -Ixz]),
        torch.stack([m * gz, zero, -m * gx, -Ixy, Iyy, -Iyz]),
        torch.stack([-m * gy, m * gx, zero, -Ixz, -Iyz, Izz])
    ]).permute(2, 0, 1)


def get_C_body(params: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    m = params[:, Param.m.value]
    gx = params[:, Param.gx.value]
    gy = params[:, Param.gy.value]
    gz = params[:, Param.gz.value]
    Ixx = params[:, Param.Ixx.value]
    Ixy = params[:, Param.Ixy.value]
    Ixz = params[:, Param.Ixz.value]
    Iyy = params[:, Param.Iyy.value]
    Iyz = params[:, Param.Iyz.value]
    Izz = params[:, Param.Izz.value]

    u = state[:, State.u.value]
    v = state[:, State.v.value]
    w = state[:, State.w.value]
    p = state[:, State.p.value]
    q = state[:, State.q.value]
    r = state[:, State.r.value]

    zero = torch.zeros(params.shape[0])

    C_12 = torch.stack([
        torch.stack([m * (gy * q + gz * r), -m * (gx * q - w), -m * (gx * r + v)]),
        torch.stack([-m * (gy * p + w), m * (gz * r + gx * p), -m * (gy * r - u)]),
        torch.stack([-m * (gz * p - v), -m * (gz * q + u), m * (gx * p + gy * q)]),
    ]).permute(2, 0, 1)

    C_22 = torch.stack([
        torch.stack([zero, -Iyz * q - Ixz * p + Izz * r, Iyz * r + Ixy * p - Iyy * q]),
        torch.stack([Ixy * q + Ixz * p - Izz * r, zero, Ixz * r + Ixy * q + Ixx * p]),
        torch.stack([-Iyz * r - Ixy * p - Iyy * q, Ixz * r + Ixy * q - Ixx * p, zero])
    ]).permute(2, 0, 1)

    return torch.cat((
        torch.cat((torch.zeros((params.shape[0], 3, 3)), C_12), dim=2),
        torch.cat((-torch.transpose(C_12, 1, 2), C_22), dim=2)
    ), dim=1)


def get_M_added(params: torch.Tensor) -> torch.Tensor:
    amu = params[:, Param.amu.value]
    amv = params[:, Param.amv.value]
    amw = params[:, Param.amw.value]
    amp = params[:, Param.amp.value]
    amq = params[:, Param.amq.value]
    amr = params[:, Param.amr.value]

    return torch.diag_embed(torch.stack([amu, amv, amw, amp, amq, amr], dim=-1))


def get_C_added(params: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    amu = params[:, Param.amu.value]
    amv = params[:, Param.amv.value]
    amw = params[:, Param.amw.value]
    amp = params[:, Param.amp.value]
    amq = params[:, Param.amq.value]
    amr = params[:, Param.amr.value]

    u = state[:, State.u.value]
    v = state[:, State.v.value]
    w = state[:, State.w.value]
    p = state[:, State.p.value]
    q = state[:, State.q.value]
    r = state[:, State.r.value]

    zero = torch.zeros(params.shape[0])

    return torch.stack([
        torch.stack([zero, zero, zero, zero, amw * w, -amv * v]),
        torch.stack([zero, zero, zero, -amw * w, zero, amu * u]),
        torch.stack([zero, zero, zero, amv * v, -amu * u, zero]),
        torch.stack([zero, amw * w, -amv * v, zero, amr * r, -amq * q]),
        torch.stack([-amw * w, zero, amu * u, -amr * r, zero, amp * p]),
        torch.stack([amv * v, -amu * u, zero, amq * q, -amp * p, zero])
    ]).permute(2, 0, 1)


def get_D(params: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    dlu = params[:, Param.dlu.value]
    dlv = params[:, Param.dlv.value]
    dlw = params[:, Param.dlw.value]
    dlp = params[:, Param.dlp.value]
    dlq = params[:, Param.dlq.value]
    dlr = params[:, Param.dlr.value]
    dqu = params[:, Param.dqu.value]
    dqv = params[:, Param.dqv.value]
    dqw = params[:, Param.dqw.value]
    dqp = params[:, Param.dqp.value]
    dqq = params[:, Param.dqq.value]
    dqr = params[:, Param.dqr.value]

    u = state[:, State.u.value]
    v = state[:, State.v.value]
    w = state[:, State.w.value]
    p = state[:, State.p.value]
    q = state[:, State.q.value]
    r = state[:, State.r.value]

    return torch.diag_embed(torch.stack([
        dlu + dqu * u,
        dlv + dqv * v,
        dlw + dqw * w,
        dlp + dqp * p,
        dlq + dqq * q,
        dlr + dqr * r,
    ], dim=-1))


def get_G(params: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    m = params[:, Param.m.value]
    v = params[:, Param.v.value]

    gx = params[:, Param.gx.value]
    gy = params[:, Param.gy.value]
    gz = params[:, Param.gz.value]
    bx = params[:, Param.bx.value]
    by = params[:, Param.by.value]
    bz = params[:, Param.bz.value]

    phi = state[:, State.phi.value]
    theta = state[:, State.theta.value]

    W = g * m
    B = g * v * rho

    return torch.stack([
        (W - B) * torch.sin(theta),
        -(W - B) * torch.cos(theta) * torch.sin(phi),
        -(W - B) * torch.cos(theta) * torch.cos(phi),
        -(gy * W - by * B) * torch.cos(theta) * torch.cos(phi) + (gz * W - bz * B) * torch.cos(theta) * torch.sin(phi),
        (gz * W - bz * B) * torch.sin(theta) + (gx * W - bx * B) * torch.cos(theta) * torch.cos(phi),
        -(gx * W - bx * B) * torch.cos(theta) * torch.sin(phi) - (gy * W - by * B) * torch.sin(theta)
    ], dim=-1)


def main():
    state = torch.zeros(2, 12)
    state[:, State.theta.value] = pi / 4

    params = torch.zeros(2, 32)
    params[:, Param.m.value] = 1
    params[:, Param.v.value] = 1
    params[:, Param.Ixx.value] = 1
    params[:, Param.Iyy.value] = 1
    params[:, Param.Izz.value] = 1

    tau = torch.zeros((2, 6))

    a = get_acceleration(params, state, tau)

    pass


if __name__ == "__main__":
    main()

