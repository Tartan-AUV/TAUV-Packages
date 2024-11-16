import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

from tauv_vision.yolact.model.config import ModelConfig


class Resnet101Backbone(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self._config = config

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # self._feature_extractor = models.feature_extraction.create_feature_extractor(resnet, return_nodes=[
        #     "layer2.3.bn2", "layer3.22.bn2", "layer4.2.bn2",
        # ])
        self._feature_extractor = models.feature_extraction.create_feature_extractor(resnet, return_nodes=[
            "layer2.1.bn2", "layer3.1.bn2", "layer4.1.bn2",
        ])

    def forward(self, img: torch.Tensor) -> (torch.Tensor, ...):
        features = self._feature_extractor(img)

        return tuple(features.values())

    @property
    def depths(self) -> (int, ...):
        return (128, 256, 512)


if __name__ == "__main__":
    config = ModelConfig(
        in_w=640,
        in_h=360,
        feature_depth=32,
        n_classes=3,
        n_prototype_masks=32,
        n_masknet_layers_pre_upsample=1,
        n_masknet_layers_post_upsample=1,
        n_classification_layers=0,
        n_box_layers=0,
        n_mask_layers=0,
        n_fpn_downsample_layers=2,
        anchor_scales=(24, 48, 96, 192, 384),
        anchor_aspect_ratios=(1 / 2, 1, 2),
        iou_pos_threshold=0.4,
        iou_neg_threshold=0.1,
        negative_example_ratio=3,
    )

    b = Resnet101Backbone(config)

    img = torch.rand((1, 3, 360, 640))
    features = b.forward(img)

    pass
