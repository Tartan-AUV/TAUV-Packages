from typing import Tuple
import random
from math import pi, floor, sin, cos

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torchvision.transforms.v2 as tfs
import matplotlib.pyplot as plt
import kornia.augmentation as A

from centernet.model.centernet import Centernet, initialize_weights, Truth
from centernet.model.backbones.dla import DLABackbone
from centernet.model.loss import gaussian_splat, loss
from centernet.model.config import ObjectConfig, ObjectConfigSet, AngleConfig, ModelConfig, TrainConfig

torch.autograd.set_detect_anomaly(True)

model_config = ModelConfig(
    in_h=512,
    in_w=512,
    backbone_heights=[2, 2, 2, 2, 2, 2],
    backbone_channels=[32, 32, 32, 32, 32, 32, 32],
    downsample_ratio=2,
    angle_bin_overlap=pi / 3,
)

train_config = TrainConfig(
    lr=1e-3,
    heatmap_focal_loss_a=2,
    heatmap_focal_loss_b=4,
    batch_size=24,
    n_batches=10000,
    loss_lambda_size=0.01,
    loss_lambda_offset=0, # Not set up to train properly
    loss_lambda_angle=0,
    loss_lambda_depth=0,
)

object_config = ObjectConfigSet(
    configs=[
        ObjectConfig(
            id="square",
            yaw=AngleConfig(
                train=True,
                modulo=pi / 2,
            ),
            pitch=AngleConfig(
                train=False,
                modulo=None,
            ),
            roll=AngleConfig(
                train=False,
                modulo=None,
            ),
            train_depth=False,
        )
    ]
)


def get_batch(model_config: ModelConfig, train_config: TrainConfig) -> Tuple[torch.Tensor, Truth]:
    heatmap = torch.zeros((train_config.batch_size, 1, model_config.out_h, model_config.out_w), dtype=torch.float32)
    img = torch.zeros((train_config.batch_size, 3, model_config.in_h, model_config.in_w), dtype=torch.float32)

    valid = torch.ones((train_config.batch_size, 1), dtype=torch.bool)
    label = torch.zeros((train_config.batch_size, 1), dtype=torch.long)

    center = ((model_config.in_h - 200) * torch.rand((train_config.batch_size, 1, 2)) + 100).to(torch.int)

    size = torch.zeros((train_config.batch_size, 1, 2), dtype=torch.float)
    size[:, :, 0] = 100 * torch.rand((train_config.batch_size, 1)) + 50
    size[:, :, 1] = size[:, :, 0]

    yaw: torch.Tensor = torch.rand((train_config.batch_size, 1)) * (pi / 2)

    for sample_i in range(train_config.batch_size):
        sample_yaw = float(yaw[sample_i, 0])
        sample_h = int(size[sample_i, 0, 0])
        sample_w = int(size[sample_i, 0, 1])
        sample_y = int(center[sample_i, 0, 0])
        sample_x = int(center[sample_i, 0, 1])

        thickness = floor(0.2 * (sample_h + sample_w) / 2)

        rot_matrix = np.array([
            [cos(sample_yaw), -sin(sample_yaw)],
            [sin(sample_yaw), cos(sample_yaw)]
        ])
        corners = np.array([
            [-sample_w / 2, -sample_h / 2],
            [sample_w / 2, -sample_h / 2],
            [sample_w / 2, sample_h / 2],
            [-sample_w / 2, sample_h / 2]
        ])

        rotated_corners = (corners @ np.transpose(rot_matrix)) + np.array([[sample_x, sample_y]])

        img_np = (255 * np.random.rand(model_config.in_h, model_config.in_w, 3)).astype(np.uint8)
        cv2.drawContours(img_np, [rotated_corners.astype(np.intp)], 0, (255, 0, 0), thickness)

        img[sample_i] = torch.from_numpy(img_np).permute(2, 0, 1).to(torch.float) / 255

        sigma = 0.05 * (sample_h + sample_w) / 2

        heatmap[sample_i] = gaussian_splat(
            model_config.out_h, model_config.out_w,
            int(sample_y // model_config.downsample_ratio), int(sample_x // model_config.downsample_ratio),
            sigma,
        )

    truth = Truth(
        heatmap=heatmap,
        valid=valid,
        label=label,
        center=center,
        size=size,
        roll=None,
        pitch=None,
        yaw=yaw,
        depth=None,
    )

    return img, truth


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"running on {device}")

    dla_backbone = DLABackbone(model_config.backbone_heights, model_config.backbone_channels)
    centernet = Centernet(dla_backbone, object_config).to(device)

    centernet.train()
    initialize_weights(centernet, [])

    optimizer = torch.optim.Adam(centernet.parameters(), lr=1e-3)

    color_transforms = A.AugmentationSequential(
        A.ColorJitter(hue=0.0, saturation=0.0, brightness=0.0),
        A.Normalize(mean=(0.51, 0.48, 0.48), std=(0.29, 0.29, 0.29))
    )

    for batch_i in range(train_config.n_batches):
        img, truth = get_batch(model_config, train_config)
        img = img.to(device)
        truth = truth.to(device)

        img_norm = color_transforms(img)

        optimizer.zero_grad()

        prediction = centernet(img_norm)

        l = loss(prediction, truth, model_config, train_config, object_config)

        print(f"loss: {float(l)}")

        l.backward()

        optimizer.step()

        if batch_i % 10 == 0:
            plt.imshow(F.sigmoid(prediction.heatmap[0, 0]).detach().cpu())
            plt.colorbar()
            plt.show()


if __name__ == "__main__":
    main()
