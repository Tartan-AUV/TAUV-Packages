import torch
import torch.nn.functional as F

from tauv_vision.yolact.model.boxes import iou_matrix


def nms(classification: torch.Tensor, box: torch.Tensor, top_k: int, iou_threshold: float, confidence_threshold: float):
    # See YOLACT fast_nms
    confidence = F.softmax(classification, dim=-1)
    max_confidence, _ = torch.max(confidence[:, :, 1:], dim=-1)

    max_confidence, idx = max_confidence.sort(dim=-1, descending=True)

    idx = idx[:, :top_k][0]
    max_confidence = max_confidence[:, :top_k][0]

    box = box[0, idx].unsqueeze(0)

    iou = iou_matrix(box, box)
    iou = torch.triu(iou, diagonal=1)

    iou_max, _ = torch.max(iou, dim=1)

    keep = (iou_max <= iou_threshold) & (max_confidence >= confidence_threshold)
    keep = keep[0]

    detection = idx[keep]

    return detection
