from centernet.model.config import ObjectConfig, ObjectConfigSet, AngleConfig, ModelConfig, TrainConfig
from math import pi

model_config = ModelConfig(
    in_h=360,
    in_w=640,
    backbone_heights=[2, 2, 2, 2, 2],
    backbone_channels=[128, 128, 128, 128, 128, 128],
    downsamples=1,
    angle_bin_overlap=pi / 3,
)

model_config_dict = model_config.to_dict()

pass

new_model_config = ModelConfig.from_dict(model_config_dict)

pass