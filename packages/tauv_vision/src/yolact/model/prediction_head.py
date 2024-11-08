import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck

from tauv_vision.yolact.model.config import ModelConfig


class PredictionHead(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self._config = config

        self._extra_layers = nn.ModuleList([
            Bottleneck(config.feature_depth, config.feature_depth // 4)
            for _ in range(self._config.n_prediction_head_layers)
        ])
        self._extra_conv_layers = nn.ModuleList([
            nn.Conv2d(config.feature_depth, config.feature_depth, kernel_size=1)
            for _ in range(self._config.n_prediction_head_layers)
        ])
        self._extra_bn_layers = nn.ModuleList([
            nn.BatchNorm2d(config.feature_depth)
            for _ in range(self._config.n_prediction_head_layers)
        ])

        self._classification_extra_layers = nn.ModuleList([
            Bottleneck(config.feature_depth, config.feature_depth // 4)
            for _ in range(self._config.n_classification_layers)
        ])
        self._classification_extra_conv_layers = nn.ModuleList([
            nn.Conv2d(config.feature_depth, config.feature_depth, kernel_size=1)
            for _ in range(self._config.n_classification_layers)
        ])
        self._classification_extra_bn_layers = nn.ModuleList([
            nn.BatchNorm2d(config.feature_depth)
            for _ in range(self._config.n_classification_layers)
        ])

        self._box_extra_layers = nn.ModuleList([
            Bottleneck(config.feature_depth, config.feature_depth // 4)
            for _ in range(self._config.n_box_layers)
        ])
        self._box_extra_conv_layers = nn.ModuleList([
            nn.Conv2d(config.feature_depth, config.feature_depth, kernel_size=1)
            for _ in range(self._config.n_box_layers)
        ])
        self._box_extra_bn_layers = nn.ModuleList([
            nn.BatchNorm2d(config.feature_depth)
            for _ in range(self._config.n_box_layers)
        ])

        self._mask_extra_layers = nn.ModuleList([
            Bottleneck(config.feature_depth, config.feature_depth // 4)
            for _ in range(self._config.n_mask_layers)
        ])
        self._mask_extra_conv_layers = nn.ModuleList([
            nn.Conv2d(config.feature_depth, config.feature_depth, kernel_size=1)
            for _ in range(self._config.n_mask_layers)
        ])
        self._mask_extra_bn_layers = nn.ModuleList([
            nn.BatchNorm2d(config.feature_depth)
            for _ in range(self._config.n_mask_layers)
        ])

        self._classification_layer = nn.Conv2d(
            config.feature_depth,
            len(config.anchor_aspect_ratios) * (config.n_classes + 1),
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self._box_encoding_layer = nn.Conv2d(
            config.feature_depth,
            len(config.anchor_aspect_ratios) * 4,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self._mask_coeff_layer = nn.Conv2d(
            config.feature_depth,
            len(config.anchor_aspect_ratios) * config.n_prototype_masks,
            kernel_size=3,
            padding=1,
            stride=1,
        )

    def forward(self, fpn_output: torch.Tensor) -> (torch.Tensor, ...):
        x = fpn_output

        for i in range(len(self._extra_layers)):
            bottleneck = self._extra_layers[i]
            conv = self._extra_conv_layers[i]
            bn = self._extra_bn_layers[i]

            x = F.relu(conv(x) + bn(bottleneck(x)))

        classification = x
        box_encoding = x
        mask_coeff = x

        for i in range(len(self._classification_extra_layers)):
            bottleneck = self._classification_extra_layers[i]
            conv = self._classification_extra_conv_layers[i]
            bn = self._classification_extra_bn_layers[i]

            classification = F.relu(conv(classification) + bn(bottleneck(classification)))

        classification = self._classification_layer(classification)
        classification = classification.permute(0, 2, 3, 1)
        classification = classification.reshape(classification.size(0), -1, self._config.n_classes + 1)

        for i in range(len(self._box_extra_layers)):
            bottleneck = self._box_extra_layers[i]
            conv = self._box_extra_conv_layers[i]
            bn = self._box_extra_bn_layers[i]

            box_encoding = F.relu(conv(box_encoding) + bn(bottleneck(box_encoding)))

        box_encoding = self._box_encoding_layer(box_encoding)
        box_encoding = box_encoding.permute(0, 2, 3, 1)
        box_encoding = box_encoding.reshape(box_encoding.size(0), -1, 4)
        # box_encoding[:, :, 0:2] = F.sigmoid(box_encoding[:, :, 0:2]) - 0.5
        # box_encoding[:, :, 0] /= x.size(2) # TODO: What does this actually do?
        # box_encoding[:, :, 1] /= x.size(3)
        # box_encoding[:, :, 2:4] = 2 * (F.sigmoid(box_encoding[:, :, 2:4]) - 0.5)
        # box_encoding[:, :, 0:2] = F.sigmoid(box_encoding[:, :, 0:2]) - 0.5

        for i in range(len(self._mask_extra_layers)):
            bottleneck = self._mask_extra_layers[i]
            conv = self._mask_extra_conv_layers[i]
            bn = self._mask_extra_bn_layers[i]

            mask_coeff = F.relu(conv(mask_coeff) + bn(bottleneck(mask_coeff)))

        mask_coeff = self._mask_coeff_layer(mask_coeff)
        mask_coeff = mask_coeff.permute(0, 2, 3, 1)
        mask_coeff = mask_coeff.reshape(mask_coeff.size(0), -1, self._config.n_prototype_masks)
        mask_coeff = F.tanh(mask_coeff)

        return classification, box_encoding, mask_coeff
