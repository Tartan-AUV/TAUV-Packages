import torch
import torch.nn as nn
import torch.nn.functional as F

from yolo_pose.model.config import Config


class FeaturePyramid(nn.Module):

    def __init__(self, in_depths: (int, ...), config: Config):
        super().__init__()

        self._config = config

        self._n_in = len(in_depths)

        self._lateral_layers = nn.ModuleList([
            nn.Conv2d(in_depth, config.feature_depth, kernel_size=1, stride=1)
            for in_depth in in_depths
        ])

        self._downsample_layers = nn.ModuleList([
            nn.Conv2d(config.feature_depth, config.feature_depth, kernel_size=3, stride=2, padding=1)
            for _ in range(config.n_fpn_downsample_layers)
        ])

        self._prediction_layers = nn.ModuleList([
            nn.Conv2d(config.feature_depth, config.feature_depth, kernel_size=3, stride=1, padding=1)
            for _ in range(self._n_in)
        ])

    def forward(self, backbone_outputs: (torch.Tensor, ...)) -> (torch.Tensor, ...):
        # Args:
        #   backbone_outputs
        #       A tuple of length len(in_depths)
        #       The ith entry is the ith output of the backbone with shape batch_size * feature_depth * in_h * in_w
        # Returns:
        #   prediction_outputs
        #       A tuple of length len(in_depths) + fpn_n_downsample_layers
        #       The ith entry is the ith output of the FPN with shape batch_size * feature_depth * w * h
        #       w and h are in_w / 4, in_h / 4 for the 0th entry and decrease by 1/2 for subsequent entries
        #       TODO: Check this

        lateral_outputs = [self._lateral_layers[i](backbone_outputs[i]) for i in range(len(backbone_outputs))]

        pyramid_outputs = [None] * self._n_in
        pyramid_outputs[-1] = lateral_outputs[-1]
        for i in range(self._n_in - 2, -1, -1):
            above_upscaled = F.interpolate(pyramid_outputs[i + 1], lateral_outputs[i].size()[2:4], mode="bilinear")
            pyramid_outputs[i] = lateral_outputs[i] + above_upscaled

        prediction_outputs = [None] * (self._n_in + self._config.n_fpn_downsample_layers)
        for i in range(self._n_in):
            prediction_outputs[i] = F.leaky_relu(self._prediction_layers[i](pyramid_outputs[i]))
        for i in range(self._config.n_fpn_downsample_layers):
            prediction_outputs[self._n_in + i] = F.leaky_relu(self._downsample_layers[i](prediction_outputs[self._n_in + i - 1]))

        return tuple(prediction_outputs)
