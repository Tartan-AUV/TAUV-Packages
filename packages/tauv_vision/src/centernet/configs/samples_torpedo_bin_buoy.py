from math import pi

from tauv_vision.centernet.model.config import ObjectConfig, ObjectConfigSet, AngleConfig, ModelConfig, TrainConfig


model_config = ModelConfig(
    in_h=360,
    in_w=640,
    backbone_heights=[2, 2, 2, 2, 2],
    backbone_channels=[128, 128, 128, 128, 128, 128],
    downsamples=2,
    angle_bin_overlap=pi / 3,
)


train_config = TrainConfig(
    lr=1e-3,
    heatmap_focal_loss_a=2,
    heatmap_focal_loss_b=4,
    heatmap_sigma_factor=0.1,
    batch_size=32,
    n_batches=0,
    n_epochs=100,
    loss_lambda_keypoint_heatmap=1.0,
    loss_lambda_keypoint_affinity=0.01,
    keypoint_heatmap_sigma=2,
    keypoint_affinity_sigma=2,
    loss_lambda_size=0.1,
    loss_lambda_offset=0.0,
    loss_lambda_angle=0.1,
    loss_lambda_depth=0.1,
    n_workers=8,
    weight_save_interval=10,
)


object_config = ObjectConfigSet(
    configs=[
        ObjectConfig(
            id="sample_24_coral",
            yaw=AngleConfig(train=False, modulo=2 * pi),
            pitch=AngleConfig(train=False, modulo=2 * pi),
            roll=AngleConfig(train=False, modulo=2 * pi),
            train_depth=False,
            train_keypoints=True,
            keypoints=[
                (0, 0, 0)
            ]
        ),
        ObjectConfig(
            id="sample_24_nautilus",
            yaw=AngleConfig(train=False, modulo=2 * pi),
            pitch=AngleConfig(train=False, modulo=2 * pi),
            roll=AngleConfig(train=False, modulo=2 * pi),
            train_depth=False,
            train_keypoints=True,
            keypoints=[
                (0, 0, 0)
            ]
        ),
        ObjectConfig(
            id="torpedo_24",
            yaw=AngleConfig(train=False, modulo=2 * pi),
            pitch=AngleConfig(train=False, modulo=2 * pi),
            roll=AngleConfig(train=False, modulo=2 * pi),
            train_depth=False,
            train_keypoints=True,
            keypoints=[
                (0, 0, 0),
            ],
        ),
        ObjectConfig(
            id="torpedo_24_octagon",
            yaw=AngleConfig(train=False, modulo=2 * pi),
            pitch=AngleConfig(train=False, modulo=2 * pi),
            roll=AngleConfig(train=False, modulo=2 * pi),
            train_depth=False,
            train_keypoints=True,
            keypoints=[
                (0, 0, 0),
            ],
        ),
        ObjectConfig(
            id="buoy_24",
            yaw=AngleConfig(train=False, modulo=2 * pi),
            pitch=AngleConfig(train=False, modulo=2 * pi),
            roll=AngleConfig(train=False, modulo=2 * pi),
            train_depth=False,
            train_keypoints=True,
            keypoints=[
                (0, 0, 0),
            ],
        ),
        ObjectConfig(
            id="bin_24",
            yaw=AngleConfig(train=False, modulo=2 * pi),
            pitch=AngleConfig(train=False, modulo=2 * pi),
            roll=AngleConfig(train=False, modulo=2 * pi),
            train_depth=False,
            train_keypoints=True,
            keypoints=[
                (0, 0, 0),
            ],
        ),
        ObjectConfig(
            id="bin_24_red",
            yaw=AngleConfig(train=False, modulo=2 * pi),
            pitch=AngleConfig(train=False, modulo=2 * pi),
            roll=AngleConfig(train=False, modulo=2 * pi),
            train_depth=False,
            train_keypoints=True,
            keypoints=[
                (0, 0, 0),
            ],
        ),
        ObjectConfig(
            id="bin_24_blue",
            yaw=AngleConfig(train=False, modulo=2 * pi),
            pitch=AngleConfig(train=False, modulo=2 * pi),
            roll=AngleConfig(train=False, modulo=2 * pi),
            train_depth=False,
            train_keypoints=True,
            keypoints=[
                (0, 0, 0),
            ],
        ),
    ]
)
