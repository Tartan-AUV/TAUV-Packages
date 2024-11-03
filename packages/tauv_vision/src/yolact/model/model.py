import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms

from tauv_vision.yolact.model.config import ModelConfig
from tauv_vision.yolact.model.weights import initialize_weights
from tauv_vision.yolact.model.backbone import Resnet101Backbone
from tauv_vision.yolact.model.feature_pyramid import FeaturePyramid
from tauv_vision.yolact.model.masknet import Masknet
from tauv_vision.yolact.model.prediction_head import PredictionHead
from tauv_vision.yolact.model.anchors import get_anchor
from tauv_vision.yolact.model.loss import loss
from tauv_vision.yolact.model.boxes import iou_matrix, box_to_mask


class Yolact(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self._backbone = Resnet101Backbone(self.config)
        self._feature_pyramid = FeaturePyramid(self._backbone.depths, self.config)
        self._masknet = Masknet(self.config)
        # self._prediction_heads = nn.ModuleList([PredictionHead(self._config) for _ in range(len(self._config.anchor_scales))])
        self._prediction_head = PredictionHead(self.config)

    def forward(self, img: torch.Tensor) -> (torch.Tensor, ...):
        backbone_outputs = self._backbone(img)

        fpn_outputs = self._feature_pyramid(backbone_outputs)

        mask_prototype = self._masknet(fpn_outputs[0])

        classifications = []
        box_encodings = []
        mask_coeffs = []
        anchors = []

        for fpn_i, fpn_output in enumerate(fpn_outputs):
            # classification, box_encoding, mask_coeff = self._prediction_heads[fpn_i](fpn_output)
            classification, box_encoding, mask_coeff = self._prediction_head(fpn_output)

            anchor = get_anchor(fpn_i, tuple(fpn_output.size()[2:4]), self.config).detach()
            anchor = anchor.to(box_encoding.device)

            classifications.append(classification)
            box_encodings.append(box_encoding)
            mask_coeffs.append(mask_coeff)
            anchors.append(anchor)

        classification = torch.cat(classifications, dim=1)
        box_encoding = torch.cat(box_encodings, dim=1)
        mask_coeff = torch.cat(mask_coeffs, dim=1)
        anchor = torch.cat(anchors, dim=1)

        return classification, box_encoding, mask_coeff, anchor, mask_prototype


def main():
    config = ModelConfig(
        in_w=640,
        in_h=360,
        feature_depth=64,
        n_classes=2,
        n_prototype_masks=16,
        n_masknet_layers_pre_upsample=1,
        n_masknet_layers_post_upsample=1,
        n_prediction_head_layers=1,
        n_classification_layers=0,
        n_box_layers=0,
        n_mask_layers=0,
        n_fpn_downsample_layers=2,
        anchor_scales=(24, 48, 96, 192, 384),
        anchor_aspect_ratios=(1 / 2, 1, 2),
        iou_pos_threshold=0.5,
        iou_neg_threshold=0.4,
        negative_example_ratio=3,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Yolact(config).to(device)
    initialize_weights(model, [model._backbone])

    # img0 = Image.open("../../img/000000.left.jpg").convert("RGB")
    # img0 = Image.open("../../img/cmu.png").convert("RGB")
    # img0 = transforms.ToTensor()(img0).to(device)
    # img1 = Image.open("../../img/000001.left.jpg").convert("RGB")
    # img1 = Image.open("../../img/cmu.png").convert("RGB")
    # img1 = transforms.ToTensor()(img1).to(device)
    # img = torch.stack((img0, img1), dim=0)

    img = torch.rand(2, 3, config.in_h, config.in_w).to(device)

    truth_valid = torch.tensor([
        [True, True],
        [True, True],
    ]).to(device)
    truth_classification = torch.tensor([
        [1, 2],
        [1, 2],
    ], dtype=torch.uint8).to(device)
    truth_box = torch.tensor([
        [
            [0.5, 0.5, 0.1, 0.1],
            [0.7, 0.7, 0.2, 0.2],
        ],
        [
            [0.6, 0.6, 0.6, 0.6],
            [0.2, 0.2, 0.7, 0.7],
        ],
    ]).to(device)
    truth_seg_map = torch.zeros(2, config.in_h, config.in_w, dtype=torch.uint8).to(device)
    for batch_i in range(truth_seg_map.size(0)):
        for detection_i in range(truth_classification.size(1)):
            # truth_seg_map[batch_i, box_to_mask(truth_box[batch_i, detection_i], (config.in_h, config.in_w)).to(torch.bool)] = truth_classification[batch_i, detection_i]
            truth_seg_map[batch_i, box_to_mask(truth_box[batch_i, detection_i], (config.in_h, config.in_w)).to(torch.bool)] = detection_i

    truth_img_valid = torch.ones((img.size(0), img.size(2), img.size(3)), dtype=torch.bool).to(device)

    truth = (truth_valid, truth_classification, truth_box, truth_seg_map, truth_img_valid)

    model.train()

    prediction = model.forward(img)

    plot_anchors(img, prediction, truth, config)

    l = loss(prediction, truth, config)
    total_loss, _ = l

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for iteration_i in range(1000):
        optimizer.zero_grad()

        prediction = model.forward(img)

        l = loss(prediction, truth, config)
        total_loss, _ = l
        print(l)

        total_loss.backward()

        optimizer.step()

    pass


def plot_anchors(img, prediction, truth, config):
    classification, box_encoding, mask_coeff, anchor, mask_prototype = prediction
    truth_valid, truth_classification, truth_box, truth_seg_map, truth_img_valid = truth

    iou = iou_matrix(anchor, truth_box)

    match_iou, match_index = torch.max(iou * truth_valid.unsqueeze(1).float(), dim=2)

    positive_match = match_iou >= config.iou_pos_threshold
    negative_match = match_iou <= config.iou_neg_threshold

    n_batch = img.size(0)

    for batch_i in range(n_batch):
        match_classification = truth_classification[batch_i, match_index[batch_i]]
        match_classification[~positive_match[batch_i]] = 0

        n_positive_match = positive_match[batch_i].sum()
        n_selected_negative_match = config.negative_example_ratio * n_positive_match

        background_confidence = F.softmax(classification[batch_i], dim=-1)[:, 0]

        _, selected_negative_match_index = torch.topk(
            torch.where(negative_match[batch_i], -background_confidence, -torch.inf),
            k=n_selected_negative_match
        )
        selected_negative_match_index = selected_negative_match_index.detach()

        selected_match = torch.clone(positive_match[batch_i])
        selected_match[selected_negative_match_index] = True


if __name__ == "__main__":
    main()