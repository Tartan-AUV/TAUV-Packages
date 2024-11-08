import torch
import torch.nn as nn
import torch.nn.functional as F

from yolo_pose.model.config import Config


class PredictionHead(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        self._config = config

        self._initial_layers = nn.Sequential(*[
            nn.Conv2d(config.feature_depth, config.feature_depth, kernel_size=3, stride=1, padding=1)
            for _ in range(self._config.n_prediction_head_layers)
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
        self._belief_coeff_layer = nn.Conv2d(
            config.feature_depth,
            len(config.anchor_aspect_ratios) * config.belief_depth * config.prototype_belief_depth,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self._affinity_coeff_layer = nn.Conv2d(
            config.feature_depth,
            len(config.anchor_aspect_ratios) * config.affinity_depth * config.prototype_affinity_depth,
            kernel_size=3,
            padding=1,
            stride=1,
        )

    def forward(self, fpn_output: torch.Tensor) -> (torch.Tensor, ...):
        x = self._initial_layers(fpn_output)

        classification = self._classification_layer(x)
        classification = classification.permute(0, 2, 3, 1)
        classification = classification.reshape(classification.size(0), -1, self._config.n_classes + 1)

        box_encoding = self._box_encoding_layer(x)
        box_encoding = box_encoding.permute(0, 2, 3, 1)
        box_encoding = box_encoding.reshape(box_encoding.size(0), -1, 4)
        box_encoding[:, :, 0:2] = F.sigmoid(box_encoding[:, :, 0:2]) - 0.5
        box_encoding[:, :, 0] /= x.size(2)
        box_encoding[:, :, 1] /= x.size(3)

        mask_coeff = self._mask_coeff_layer(x)
        mask_coeff = mask_coeff.permute(0, 2, 3, 1)
        mask_coeff = mask_coeff.reshape(mask_coeff.size(0), -1, self._config.n_prototype_masks)
        mask_coeff = F.tanh(mask_coeff)

        belief_coeff = self._belief_coeff_layer(x)
        belief_coeff = belief_coeff.permute(0, 2, 3, 1)
        belief_coeff = belief_coeff.reshape(belief_coeff.size(0), -1, self._config.belief_depth, self._config.prototype_belief_depth)
        belief_coeff = F.tanh(belief_coeff)

        affinity_coeff = self._affinity_coeff_layer(x)
        affinity_coeff = affinity_coeff.permute(0, 2, 3, 1)
        affinity_coeff = affinity_coeff.reshape(affinity_coeff.size(0), -1, self._config.affinity_depth, self._config.prototype_affinity_depth)
        affinity_coeff = F.tanh(affinity_coeff)

        return classification, box_encoding, mask_coeff, belief_coeff, affinity_coeff
