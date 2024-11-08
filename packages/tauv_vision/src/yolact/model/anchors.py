import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from tauv_vision.yolact.model.config import ModelConfig


def get_anchor(fpn_i: int, fpn_size: (int, int), config: ModelConfig):
    # y, x, h, w

    n_anchors = len(config.anchor_aspect_ratios)

    y = (torch.arange(0, fpn_size[0]) + 0.5) / fpn_size[0]
    x = (torch.arange(0, fpn_size[1]) + 0.5) / fpn_size[1]
    y, x = torch.meshgrid(y, x, indexing='ij')
    y = y.flatten()
    x = x.flatten()
    y = torch.tile(y, (1, n_anchors))
    x = torch.tile(x, (1, n_anchors))

    hs = []
    ws = []
    scale = config.anchor_scales[fpn_i]
    for aspect_ratio in config.anchor_aspect_ratios:
        in_size = (config.in_h + config.in_w) / 2
        h = (scale / in_size) * sqrt(aspect_ratio)
        w = (scale / in_size) / sqrt(aspect_ratio)

        h = torch.full((1, fpn_size[0] * fpn_size[1]), h)
        w = torch.full((1, fpn_size[0] * fpn_size[1]), w)

        hs.append(h)
        ws.append(w)

    h = torch.cat(hs, dim=-1)
    w = torch.cat(ws, dim=-1)

    anchor = torch.stack((y, x, h, w), dim=1).permute(0, 2, 1)

    return anchor