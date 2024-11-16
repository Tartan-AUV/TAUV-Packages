import torch
import torch.nn as nn
import torch.nn.functional as F

from tauv_vision.yolact.model.config import ModelConfig


class Masknet(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self._config = config

        self._layers_1 = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(config.feature_depth, config.feature_depth, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
            )
            for _ in range(1)
        ])

        self._upsample_layer_1 = nn.ConvTranspose2d(config.feature_depth, config.feature_depth, 3, stride=2, padding=1)

        self._layers_2 = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(config.feature_depth, config.feature_depth, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
            )
            for _ in range(1)
        ])

        self._upsample_layer_2 = nn.ConvTranspose2d(config.feature_depth, config.feature_depth, 3, stride=2, padding=1)

        self._layers_3 = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(config.feature_depth, config.feature_depth, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
            )
            for _ in range(1)
        ])

        self._output_layer = nn.Conv2d(config.feature_depth, config.n_prototype_masks, kernel_size=1, stride=1)

    def forward(self, fpn_output: torch.Tensor) -> torch.Tensor:
        x = self._layers_1(fpn_output)
        x = self._upsample_layer_1(x, output_size=(x.size(0), x.size(1), 2 * x.size(2), 2 * x.size(3)))
        x = F.leaky_relu(x)
        x = self._layers_2(x)
        x = self._upsample_layer_2(x, output_size=(x.size(0), x.size(1), 2 * x.size(2), 2 * x.size(3)))
        x = F.leaky_relu(x)
        x = self._layers_3(x)
        x = self._output_layer(x)
        x = F.leaky_relu(x)

        return x