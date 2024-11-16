from dataclasses import dataclass, asdict
import json
import pathlib
from typing import Optional


@dataclass
class ModelConfig:
    in_w: int
    in_h: int

    feature_depth: int

    n_classes: int
    n_prototype_masks: int

    n_masknet_layers_pre_upsample: int
    n_masknet_layers_post_upsample: int

    n_prediction_head_layers: int
    n_classification_layers: int
    n_box_layers: int
    n_mask_layers: int

    n_fpn_downsample_layers: int

    anchor_scales: (int, ...)
    anchor_aspect_ratios: (int, ...)

    box_variances: (float, float)

    iou_pos_threshold: float
    iou_neg_threshold: float

    negative_example_ratio: int

    img_mean: (float, float, float)
    img_stddev: (float, float, float)

    def save(self, path: pathlib.Path):
        with open(path, "w") as fp:
            json.dump(asdict(self), fp, indent=2)

    @classmethod
    def load(cls, path: pathlib.Path):
        with open(path, "r") as fp:
            data = json.load(fp)
        return cls(**data)


@dataclass
class TrainConfig:
    lr: float
    momentum: float
    weight_decay: float
    grad_max_norm: float

    n_epochs: int
    batch_size: int
    epoch_n_batches: int

    weight_save_interval: int
    gradient_save_frequency: int

    channel_shuffle_p: float

    color_jitter_p: float
    color_jitter_brightness: float
    color_jitter_contrast: float
    color_jitter_saturation: float
    color_jitter_hue: float

    gaussian_noise_p: float
    gaussian_noise_var_limit: (float, float)

    horizontal_flip_p: float
    vertical_flip_p: float

    blur_limit: (int, int)
    blur_p: float

    ssr_p: float
    ssr_shift_limit: (float, float)
    ssr_scale_limit: (float, float)
    ssr_rotate_limit: (float, float)

    perspective_p: float
    perspective_scale_limit: (float, float)

    min_visibility: float

    n_workers: int

    def save(self, path: pathlib.Path):
        with open(path, "w") as fp:
            json.dump(asdict(self), fp, indent=2)

    @classmethod
    def load(cls, path: pathlib.Path):
        with open(path, "r") as fp:
            data = json.load(fp)
        return cls(**data)


@dataclass
class ClassConfig:
    id: str
    index: int # starts at 1


@dataclass
class ClassConfigSet:

    configs: [ClassConfig]

    def get_by_index(self, index: int) -> Optional[ClassConfig]:
        for config in self.configs:
            if config.index == index:
                return config

        return None

    def save(self, path: pathlib.Path):
        with open(path, "w") as fp:
            json.dump(asdict(self), fp, indent=2)

    @classmethod
    def load(cls, path: pathlib.Path):
        with open(path, "r") as fp:
            data = json.load(fp)

        return cls([ClassConfig(d["id"], d["index"]) for d in data["configs"]])
