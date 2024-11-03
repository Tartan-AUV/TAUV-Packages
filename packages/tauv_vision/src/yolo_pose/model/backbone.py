import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

from yolo_pose.model.config import Config


class Resnet101Backbone(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        self._config = config

        resnet_101 = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

        self._feature_extractor = models.feature_extraction.create_feature_extractor(resnet_101, return_nodes=[
            "layer2.3.conv2", "layer3.22.conv2", "layer4.2.conv2"
        ])

    def forward(self, img: torch.Tensor) -> (torch.Tensor, ...):
        features = self._feature_extractor(img)

        return tuple(features.values())

    @property
    def depths(self) -> (int, ...):
        return (128, 256, 512)


if __name__ == "__main__":
    b = Resnet101Backbone(Config())