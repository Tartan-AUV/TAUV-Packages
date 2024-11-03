import torch
from math import pi, floor
import torch.nn.functional as F
from enum import Enum
from dataclasses import dataclass
import matplotlib.pyplot as plt

from tauv_vision.centernet.model.centernet import Prediction
from tauv_vision.centernet.model.decode import angle_get_bins
from tauv_vision.centernet.model.config import ModelConfig, TrainConfig, ObjectConfig, ObjectConfigSet, AngleConfig
from tauv_vision.datasets.load.pose_dataset import PoseSample


@dataclass
class Losses:
    total: torch.Tensor = torch.zeros(1)
    heatmap: torch.Tensor = torch.zeros(1)
    keypoint_heatmap: torch.Tensor = torch.zeros(1)
    keypoint_affinity: torch.Tensor = torch.zeros(1)
    offset: torch.Tensor = torch.zeros(1)
    size: torch.Tensor = torch.zeros(1)
    roll: torch.Tensor = torch.zeros(1)
    pitch: torch.Tensor = torch.zeros(1)
    yaw: torch.Tensor = torch.zeros(1)
    depth: torch.Tensor = torch.zeros(1)

    avg_size_error: torch.Tensor = torch.zeros(1)
    max_size_error: torch.Tensor = torch.zeros(1)


def generate_heatmap(truth: PoseSample, model_config: ModelConfig, train_config: TrainConfig, object_config: ObjectConfigSet) -> torch.Tensor:
    # result is [batch_size, n_labels, out_h, out_w]

    batch_size, n_objects = truth.valid.shape
    n_labels = object_config.n_labels
    out_h = model_config.out_h
    out_w = model_config.out_w

    heatmap = torch.zeros((batch_size, n_labels, out_h, out_w), dtype=torch.float32, device=truth.valid.device)

    y = torch.arange(0, out_h, device=truth.valid.device)
    x = torch.arange(0, out_w, device=truth.valid.device)

    y, x = torch.meshgrid(y, x, indexing="ij")

    for sample_i in range(batch_size):
        for object_i in range(n_objects):
            if not truth.valid[sample_i, object_i]:
                continue

            cy = floor(truth.center[sample_i, object_i, 0] * model_config.in_h / model_config.downsample_ratio)
            cx = floor(truth.center[sample_i, object_i, 1] * model_config.in_w / model_config.downsample_ratio)

            # h = float(truth.size[sample_i, object_i, 0] * model_config.in_h)
            # w = float(truth.size[sample_i, object_i, 1] * model_config.in_w)

            # sigma = train_config.heatmap_sigma_factor * (h + w) / 2
            sigma = train_config.keypoint_heatmap_sigma

            if sigma < 0.1:
                print("tiny sigma!")
                sigma = 0.1

            heatmap[sample_i, truth.label[sample_i, object_i]] = torch.maximum(
                    heatmap[sample_i, truth.label[sample_i, object_i]],
                    torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
            )

    # TODO: get rid of this, this really shouldn't be necessary
    heatmap = torch.nan_to_num(heatmap)

    return heatmap


def generate_keypoint_heatmap(truth: PoseSample, model_config: ModelConfig, train_config: TrainConfig, object_config: ObjectConfigSet) -> torch.Tensor:
    # result is [batch_size, n_labels, out_h, out_w]

    batch_size, n_keypoint_instances = truth.keypoint_valid.shape
    n_keypoints = object_config.n_keypoints
    out_h = model_config.out_h
    out_w = model_config.out_w

    heatmap = torch.zeros((batch_size, n_keypoints, out_h, out_w), dtype=torch.float32, device=truth.valid.device)
    affinity_weight = torch.zeros((batch_size, n_keypoints, out_h, out_w), dtype=torch.float32, device=truth.valid.device)
    affinity = torch.zeros((batch_size, n_keypoints, 2, out_h, out_w), dtype=torch.float32, device=truth.valid.device)

    distance = torch.full((batch_size, n_keypoints, out_h, out_w), fill_value=torch.inf, dtype=torch.float32, device=truth.valid.device)

    y = torch.arange(0, out_h, device=truth.valid.device)
    x = torch.arange(0, out_w, device=truth.valid.device)

    y, x = torch.meshgrid(y, x, indexing="ij")

    for sample_i in range(batch_size):
        for keypoint_instance_i in range(n_keypoint_instances):
            if not truth.keypoint_valid[sample_i, keypoint_instance_i]:
                continue

            keypoint_i = truth.keypoint_label[sample_i, keypoint_instance_i]

            cy = floor(truth.keypoint_center[sample_i, keypoint_instance_i, 0] * model_config.in_h / model_config.downsample_ratio)
            cx = floor(truth.keypoint_center[sample_i, keypoint_instance_i, 1] * model_config.in_w / model_config.downsample_ratio)

            heatmap[sample_i, keypoint_i] = torch.maximum(
                heatmap[sample_i, keypoint_i],
                torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * train_config.keypoint_heatmap_sigma ** 2))
            )

            affinity_weight[sample_i, keypoint_i] = torch.maximum(
                affinity_weight[sample_i, keypoint_i],
                torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * train_config.keypoint_affinity_sigma ** 2))
            )

            keypoint_displacement = torch.stack((y / model_config.out_h, x / model_config.out_w), dim=0) - truth.center[sample_i, truth.keypoint_object_index[sample_i, keypoint_instance_i]].unsqueeze(1).unsqueeze(2)

            keypoint_displacement = torch.nan_to_num(keypoint_displacement, 0)

            keypoint_distance = torch.nan_to_num(torch.sqrt(keypoint_displacement[0] ** 2 + keypoint_displacement[1] ** 2), 1)

            keypoint_affinity = keypoint_displacement / keypoint_distance

            affinity[sample_i, keypoint_i] = torch.where(
                keypoint_distance < distance[sample_i, keypoint_i],
                keypoint_affinity,
                affinity[sample_i, keypoint_i],
            )

            distance[sample_i, keypoint_i] = torch.min(distance[sample_i, keypoint_i], keypoint_distance)

    # TODO: get rid of this, this really shouldn't be necessary
    heatmap = torch.nan_to_num(heatmap)
    affinity_weight = torch.nan_to_num(affinity_weight)
    affinity = torch.nan_to_num(affinity)

    return heatmap, affinity_weight, affinity


def out_index_for_position(position: torch.Tensor, model_config: ModelConfig) -> torch.Tensor:
    return torch.stack((
        torch.clamp(((position[:, :, 0] * model_config.in_h) / model_config.downsample_ratio).to(torch.long), 0, model_config.out_h - 1),
        torch.clamp(((position[:, :, 1] * model_config.in_w) / model_config.downsample_ratio).to(torch.long), 0, model_config.out_w - 1),
    ), dim=-1)


class Angle(Enum):
    Roll = 0
    Pitch = 1
    Yaw = 2


def angle_range(truth_label: torch.Tensor, object_config: ObjectConfigSet, angle: Angle) -> torch.Tensor:
    # truth_label is [batch_size, n_objects]
    #
    # result is [batch_size, n_objects]

    batch_size, n_objects = truth_label.shape
    device = truth_label.device

    result = torch.zeros_like(truth_label, dtype=torch.float32)

    for sample_i in range(batch_size):
        for object_i in range(n_objects):
            label = truth_label[sample_i, object_i]

            if angle == Angle.Roll:
                modulo = object_config.configs[label].roll.modulo
            elif angle == Angle.Pitch:
                modulo = object_config.configs[label].pitch.modulo
            elif angle == Angle.Yaw:
                modulo = object_config.configs[label].yaw.modulo

            if modulo is not None:
                result[sample_i, object_i] = modulo

    return result.to(device)


def loss(prediction: Prediction, truth: PoseSample, model_config: ModelConfig, train_config: TrainConfig, object_config: ObjectConfigSet, img) -> Losses:

    losses = Losses()

    heatmap = generate_heatmap(truth, model_config, train_config, object_config)

    if prediction.keypoint_heatmap is not None:
        keypoint_heatmap, keypoint_affinity_weight, keypoint_affinity = generate_keypoint_heatmap(truth, model_config, train_config, object_config)

    out_index = out_index_for_position(truth.center, model_config) # [batch_size, n_objects, 2]

    batch_size, n_objects, _ = out_index.shape
    device = out_index.device

    # TODO: Vectorize this
    prediction_size = torch.zeros((batch_size, n_objects, 2), dtype=torch.float32, device=device) # [batch_size, n_objects, 2]
    prediction_offset = torch.zeros((batch_size, n_objects, 2), dtype=torch.float32, device=device) # [batch_size, n_objects, 2]

    if prediction.roll_bin is not None:
        prediction_roll_bin = torch.zeros((batch_size, n_objects, 4), dtype=torch.float32, device=device)
        prediction_roll_offset = torch.zeros((batch_size, n_objects, 4), dtype=torch.float32, device=device)

    if prediction.pitch_bin is not None:
        prediction_pitch_bin = torch.zeros((batch_size, n_objects, 4), dtype=torch.float32, device=device)
        prediction_pitch_offset = torch.zeros((batch_size, n_objects, 4), dtype=torch.float32, device=device)

    if prediction.yaw_bin is not None:
        prediction_yaw_bin = torch.zeros((batch_size, n_objects, 4), dtype=torch.float32, device=device)
        prediction_yaw_offset = torch.zeros((batch_size, n_objects, 4), dtype=torch.float32, device=device)

    if prediction.depth is not None:
        prediction_depth = torch.zeros((batch_size, n_objects), dtype=torch.float32, device=device)

    for sample_i in range(batch_size):
        for object_i in range(n_objects):
            prediction_size[sample_i, object_i] = prediction.size[sample_i, out_index[sample_i, object_i, 0], out_index[sample_i, object_i, 1]]
            prediction_offset[sample_i, object_i] = prediction.offset[sample_i, out_index[sample_i, object_i, 0], out_index[sample_i, object_i, 1]]

            if prediction.roll_bin is not None:
                prediction_roll_bin[sample_i, object_i] = prediction.roll_bin[sample_i, out_index[sample_i, object_i, 0], out_index[sample_i, object_i, 1]]
                prediction_roll_offset[sample_i, object_i] = prediction.roll_offset[sample_i, out_index[sample_i, object_i, 0], out_index[sample_i, object_i, 1]]

            if prediction.pitch_bin is not None:
                prediction_pitch_bin[sample_i, object_i] = prediction.pitch_bin[sample_i, out_index[sample_i, object_i, 0], out_index[sample_i, object_i, 1]]
                prediction_pitch_offset[sample_i, object_i] = prediction.pitch_offset[sample_i, out_index[sample_i, object_i, 0], out_index[sample_i, object_i, 1]]

            if prediction.yaw_bin is not None:
                prediction_yaw_bin[sample_i, object_i] = prediction.yaw_bin[sample_i, out_index[sample_i, object_i, 0], out_index[sample_i, object_i, 1]]
                prediction_yaw_offset[sample_i, object_i] = prediction.yaw_offset[sample_i, out_index[sample_i, object_i, 0], out_index[sample_i, object_i, 1]]

            if prediction.depth is not None:
                prediction_depth[sample_i, object_i] = prediction.depth[sample_i, out_index[sample_i, object_i, 0], out_index[sample_i, object_i, 1]]

    n_valid = min(int(truth.valid.to(torch.float32).sum()), 1)

    l_heatmap = focal_loss(F.sigmoid(prediction.heatmap), heatmap, alpha=train_config.heatmap_focal_loss_a, beta=train_config.heatmap_focal_loss_b)
    l_heatmap = l_heatmap.sum()
    losses.heatmap = l_heatmap
    l = l_heatmap

    if prediction.keypoint_heatmap is not None:
        l_keypoint_heatmap = focal_loss(F.sigmoid(prediction.keypoint_heatmap), keypoint_heatmap, alpha=train_config.heatmap_focal_loss_a, beta=train_config.heatmap_focal_loss_b)
        l_keypoint_heatmap = train_config.loss_lambda_keypoint_heatmap * l_keypoint_heatmap.sum()
        losses.keypoint_heatmap = l_keypoint_heatmap
        l += l_keypoint_heatmap

    if prediction.keypoint_affinity is not None:
        l_keypoint_affinity = F.mse_loss(prediction.keypoint_affinity, keypoint_affinity, reduction="none")
        l_keypoint_affinity = train_config.loss_lambda_keypoint_affinity * (keypoint_affinity_weight.unsqueeze(2) * l_keypoint_affinity).sum()
        losses.keypoint_affinity = l_keypoint_affinity
        l += l_keypoint_affinity

    # truth_pixel_size = truth.size * torch.Tensor((model_config.in_h, model_config.in_w)).to(truth.size.device).unsqueeze(0).unsqueeze(1)
    l_size = F.l1_loss(prediction_size, truth.size, reduction="none")
    l_size = train_config.loss_lambda_size * (truth.valid.unsqueeze(-1) * l_size).sum() / n_valid
    losses.size = l_size
    l += l_size

    size_error = torch.where(truth.valid.unsqueeze(-1), torch.abs(prediction_size - truth.size), torch.nan)
    avg_size_error = size_error.nanmean()
    max_size_error = torch.where(torch.isnan(size_error), 0, size_error).max()

    losses.avg_size_error = avg_size_error
    losses.max_size_error = max_size_error

    truth_pixel_center = truth.center * torch.Tensor((model_config.in_h, model_config.in_w)).to(truth.center.device).unsqueeze(0).unsqueeze(1)
    truth_pixel_offset = truth_pixel_center - model_config.downsample_ratio * (truth_pixel_center / model_config.downsample_ratio).to(torch.long)
    l_offset = F.l1_loss(prediction_offset, truth_pixel_offset, reduction="none")
    l_offset = train_config.loss_lambda_offset * (truth.valid.unsqueeze(-1) * l_offset).sum() / n_valid
    losses.offset = l_offset
    l += l_offset

    if prediction.roll_bin is not None:
        roll_theta_range = angle_range(truth.label, object_config, Angle.Roll)
        l_roll = angle_loss(prediction_roll_bin, prediction_roll_offset, truth.roll, roll_theta_range, model_config.angle_bin_overlap).sum()
        l_roll = train_config.loss_lambda_angle * (truth.valid * l_roll).sum() / n_valid
        losses.roll = l_roll
        l += l_roll

    if prediction.pitch_bin is not None:
        pitch_theta_range = angle_range(truth.label, object_config, Angle.Pitch)
        l_pitch = angle_loss(prediction_pitch_bin, prediction_pitch_offset, truth.pitch, pitch_theta_range, model_config.angle_bin_overlap).sum()
        l_pitch = train_config.loss_lambda_angle * (truth.valid * l_pitch).sum() / n_valid
        losses.pitch = l_pitch
        l += l_pitch

    if prediction.yaw_bin is not None:
        yaw_theta_range = angle_range(truth.label, object_config, Angle.Yaw)
        l_yaw = angle_loss(prediction_yaw_bin, prediction_yaw_offset, truth.yaw, yaw_theta_range, model_config.angle_bin_overlap).sum()
        l_yaw = train_config.loss_lambda_angle * (truth.valid * l_yaw).sum() / n_valid
        losses.yaw = l_yaw
        l += l_yaw

    if prediction.depth is not None:
        l_depth = depth_loss(prediction_depth, truth.depth)
        l_depth = train_config.loss_lambda_depth * (truth.valid * l_depth).sum() / n_valid
        losses.depth = l_depth
        l += l_depth

    losses.total = l

    return losses


def focal_loss(prediction: torch.Tensor, truth: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # prediction is [batch_size, n_classes, h, w]
    # truth is [batch_size, n_classes, h, w]

    p = torch.isclose(truth, torch.Tensor([1]).to(truth.device))
    N = torch.sum(p)

    loss_p = ((1 - prediction) ** alpha) * torch.log(torch.clamp(prediction, min=1e-4)) * p.float()
    loss_n = ((1 - truth) ** beta) * (prediction ** alpha) * torch.log(torch.clamp(1 - prediction, min=1e-4)) * (1 - p.float())

    if N == 0:
        loss = -loss_p
    else:
        loss = -(loss_p + loss_n) / N

    return loss


def angle_in_range(angles: torch.Tensor, range_min: float, range_max: float) -> torch.Tensor:
    # angles is [shape]
    #
    # result is [shape]

    range_min = range_min % (2 * pi)
    range_max = range_max % (2 * pi)
    angles = angles % (2 * pi)
    if range_min < range_max:
        return (range_min <= angles) & (angles <= range_max)
    else:
        return (range_min <= angles) | (angles <= range_max)


def angle_loss(predicted_bin: torch.Tensor, predicted_offset: torch.Tensor, truth: torch.Tensor, theta_range: torch.Tensor, bin_overlap: float) -> torch.Tensor:
    # All angles are taken to fall in range [0, theta_range)
    #
    # predicted_bin is [batch_size, n_detections, 4]
    #   predicted_bin[0:2] are [outside, inside] classifications for bin 0
    #   predicted_bin[2:4] are [outside, inside] classifications for bin 1
    # predicted_offset is [batch_size, n_detections, 4]
    #   predicted_offset[0:2] are [sin, cos] offsets for bin 0
    #   predicted_offset[2:4] are [sin, cos] offsets for bin 1
    #
    # truth is [batch_size, n_detections]
    #
    # theta_range is [batch_size, n_detections]
    #
    # result is [batch_size, n_detections]

    batch_size, n_detections = truth.shape

    truth = truth % theta_range              # angles from [0, theta_range)
    truth = truth * (2 * pi / theta_range)   # angles from [0, 2 * pi)

    bins = angle_get_bins(bin_overlap)
    (bin_0_center, bin_0_min, bin_0_max), (bin_1_center, bin_1_min, bin_1_max) = bins

    inside_bin_0 = angle_in_range(truth, bin_0_min, bin_0_max).to(torch.long)  # [batch_size, n_detections]
    inside_bin_1 = angle_in_range(truth, bin_1_min, bin_1_max).to(torch.long)  # [batch_size, n_detections]

    offsets_bin_0 = torch.stack((torch.sin(truth - bin_0_center), torch.cos(truth - bin_0_center)), dim=-1) # [batch_size, n_detections, 2]
    offsets_bin_1 = torch.stack((torch.sin(truth - bin_1_center), torch.cos(truth - bin_1_center)), dim=-1) # [batch_size, n_detections, 2]

    classification_loss_bin_0 = F.cross_entropy(predicted_bin[:, :, 0:2].reshape(-1, 2), inside_bin_0.reshape(-1), reduction="none") # [batch_size x n_detections]
    classification_loss_bin_1 = F.cross_entropy(predicted_bin[:, :, 2:4].reshape(-1, 2), inside_bin_1.reshape(-1), reduction="none") # [batch_size x n_detections]

    offset_loss_bin_0 = F.l1_loss(predicted_offset[:, :, 0:2].reshape(-1, 2), offsets_bin_0.reshape(-1, 2), reduction="none").sum(dim=-1) # [batch_size x n_detections]
    offset_loss_bin_1 = F.l1_loss(predicted_offset[:, :, 2:4].reshape(-1, 2), offsets_bin_1.reshape(-1, 2), reduction="none").sum(dim=-1) # [batch_size x n_detections]

    result = (classification_loss_bin_0 + classification_loss_bin_1
              + inside_bin_0.reshape(-1).to(torch.float) * offset_loss_bin_0
              + inside_bin_1.reshape(-1).to(torch.float) * offset_loss_bin_1)   # [batch_size x n_detections]

    result = result.reshape(batch_size, n_detections)   # [batch_size, n_detections]

    return result


def depth_loss(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    # prediction is [batch_size, n_detections]
    # truth is [batch_size, n_detections]
    #
    # result is [batch_size, n_detections]

    batch_size, n_detections = truth.shape

    result = F.l1_loss(1 / F.sigmoid(prediction.reshape(-1)) - 1, truth.reshape(-1), reduction="none")
    result = result.reshape(batch_size, n_detections)

    return result
