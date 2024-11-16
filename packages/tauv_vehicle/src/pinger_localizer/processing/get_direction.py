import numpy as np
from math import atan2, sqrt

def get_direction(delays, positions):
    relative_positions = positions[1:, :] - positions[0, :]

    direction = np.linalg.inv(relative_positions) @ -delays

    direction = direction / np.linalg.norm(direction)

    psi = atan2(direction[1], direction[0])
    theta = -atan2(direction[2], sqrt(direction[0] ** 2 + direction[1] ** 2))

    return direction, psi, theta
