from typing import Optional, Dict, List, Tuple

from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    backbone_heights: [int]
    backbone_channels: [int]

    in_h: int
    in_w: int

    downsamples: int

    angle_bin_overlap: float

    @property
    def out_h(self) -> int:
        return self.in_h // self.downsample_ratio

    @property
    def out_w(self) -> int:
        return self.in_w // self.downsample_ratio

    @property
    def downsample_ratio(self) -> int:
        return 2 ** self.downsamples

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class TrainConfig:
    lr: float

    batch_size: int
    n_batches: int
    n_epochs: int

    heatmap_focal_loss_a: float
    heatmap_focal_loss_b: float
    heatmap_sigma_factor: float

    keypoint_heatmap_sigma: float
    keypoint_affinity_sigma: float

    loss_lambda_keypoint_heatmap: float
    loss_lambda_keypoint_affinity: float
    loss_lambda_size: float
    loss_lambda_offset: float
    loss_lambda_angle: float
    loss_lambda_depth: float

    n_workers: int

    weight_save_interval: int

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class AngleConfig:
    train: bool
    modulo: Optional[float]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class ObjectConfig:
    id: str

    yaw: AngleConfig
    pitch: AngleConfig
    roll: AngleConfig

    train_depth: bool

    train_keypoints: bool

    keypoints: Optional[List[Tuple[float, float, float]]]

    def to_dict(self):
        return {
            "id": self.id,
            "yaw": self.yaw.to_dict(),
            "pitch": self.pitch.to_dict(),
            "roll": self.roll.to_dict(),
            "train_depth": self.train_depth,
            "train_keypoints": self.train_keypoints,
            "keypoints": [list(keypoint) for keypoint in self.keypoints] if self.keypoints is not None else None,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data["id"],
            yaw=AngleConfig.from_dict(data["yaw"]),
            pitch=AngleConfig.from_dict(data["pitch"]),
            roll=AngleConfig.from_dict(data["roll"]),
            train_depth=data["train_depth"],
            train_keypoints=data["train_keypoints"],
            keypoints=[tuple(keypoint) for keypoint in data["keypoints"]] if data["keypoints"] is not None else None,
        )



class ObjectConfigSet:

    def __init__(self, configs: [ObjectConfig]):
        self.configs: [ObjectConfig] = configs

        keypoint_index_encode: Dict[Tuple[int, int], int] = {}
        keypoint_index_decode: Dict[int, Tuple[int, int]] = {}

        keypoint_index = 0
        for object_index, config in enumerate(self.configs):
            if config.keypoints is None:
                continue

            for object_keypoint_index, _ in enumerate(config.keypoints):
                keypoint_index_encode[(object_index, object_keypoint_index)] = keypoint_index
                keypoint_index_decode[keypoint_index] = (object_index, object_keypoint_index)

                keypoint_index += 1

        self._keypoint_index_encode = keypoint_index_encode
        self._keypoint_index_decode = keypoint_index_decode

    def to_dict(self):
        return {
            "object_configs": [object_config.to_dict() for object_config in self.configs]
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            configs=[ObjectConfig.from_dict(object_config) for object_config in data["object_configs"]]
        )

    @property
    def train_yaw(self) -> bool:
        return any([config.yaw.train for config in self.configs])

    @property
    def train_pitch(self) -> bool:
        return any([config.pitch.train for config in self.configs])

    @property
    def train_roll(self) -> bool:
        return any([config.roll.train for config in self.configs])

    @property
    def train_depth(self) -> bool:
        return any([config.train_depth for config in self.configs])

    @property
    def train_keypoints(self) -> bool:
        return any([config.train_keypoints for config in self.configs])

    @property
    def n_labels(self) -> int:
        return len(self.configs)

    @property
    def n_keypoints(self) -> int:
        return sum([len(config.keypoints) if config.keypoints is not None else 0 for config in self.configs])

    @property
    def label_id_to_index(self) -> Dict[str, int]:
        return {config.id: i for (i, config) in enumerate(self.configs)}

    def encode_keypoint_index(self, object_index: int, object_keypoint_index: int) -> int:
        return self._keypoint_index_encode[(object_index, object_keypoint_index)]

    def decode_keypoint_index(self, keypoint_index) -> Tuple[int, int]:
        return self._keypoint_index_decode[keypoint_index]

    def get_by_label(self, label: str) -> ObjectConfig:
        return self.configs[self.label_id_to_index[label]]
