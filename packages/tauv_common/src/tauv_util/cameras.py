from dataclasses import dataclass
import numpy as np


@dataclass
class CameraIntrinsics:
    f_x: float
    f_y: float
    c_x: float
    c_y: float

    @classmethod
    def from_matrix(cls, k: np.array):
        if k.shape == (9,):
            return cls(k[0], k[4], k[2], k[5])
        elif k.shape == (3, 3):
            return cls(k[0, 0], k[1, 1], k[0, 2], k[1, 2])
        else:
            raise ValueError(f"unsupported shape: {k.shape}")

    def to_matrix(self) -> np.array:
        return np.array([
            [self.f_x, 0, self.c_x],
            [0, self.f_y, self.c_y],
            [0, 0, 1],
        ]).astype(float)


@dataclass
class CameraDistortion:
    d: np.array

    @classmethod
    def from_matrix(cls, d: np.array):
        if d.shape == (5,):
            return cls(d.copy())
        else:
            raise ValueError(f"unsupported shape: {d.shape}")

    def to_matrix(self) -> np.array:
        return self.d.copy()