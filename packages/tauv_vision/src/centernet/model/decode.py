from typing import Tuple, Optional, List
from dataclasses import dataclass
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
from math import pi, atan2
import torch.nn.functional as F
from spatialmath import SE3, SO3

from tauv_vision.centernet.model.centernet import Prediction
from tauv_vision.centernet.model.config import ModelConfig, ObjectConfigSet


@dataclass
class Detection:
    label: int
    score: float
    y: float
    x: float
    h: float
    w: float

    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None

    depth: Optional[float] = None


@dataclass
class KeypointDetection:
    label: int
    score: float

    y: float
    x: float

    w: float
    h: float
    depth: float

    keypoints: List[Optional[Tuple[float, float, float]]]
    keypoint_scores: List[Optional[float]]
    keypoint_affinities: List[Optional[Tuple[float, float, float]]]

    cam_t_object: SE3


def decode_keypoints(prediction: Prediction,
                     model_config: ModelConfig, object_config: ObjectConfigSet,
                     M_projection: np.array,
                     n_detections: int, keypoint_n_detections: int,
                     score_threshold: float, keypoint_score_threshold: float, keypoint_angle_threshold: float) -> [[KeypointDetection]]:
    heatmap = F.sigmoid(prediction.heatmap)
    heatmap = heatmap_nms(heatmap, kernel_size=3)
    detected_index, detected_label, detected_score = heatmap_detect(heatmap, n_detections)

    keypoint_heatmap = F.sigmoid(prediction.keypoint_heatmap)
    keypoint_heatmap = heatmap_nms(keypoint_heatmap, kernel_size=3)
    detected_keypoint_index, detected_keypoint_label, detected_keypoint_score = heatmap_detect(keypoint_heatmap, keypoint_n_detections)

    if prediction.depth is not None:
        depth = 1 / F.sigmoid(prediction.depth)

    batch_size = detected_index.shape[0]

    detections = []

    for sample_i in range(batch_size):
        sample_detections = []

        for detection_i in range(n_detections):
            if detected_score[sample_i, detection_i] < score_threshold:
                break

            label = int(detected_label[sample_i, detection_i])

            config = object_config.configs[label]

            n_keypoints = len(config.keypoints)

            detection = KeypointDetection(
                label=label,
                score=float(detected_score[sample_i, detection_i]),
                y=float(detected_index[sample_i, detection_i, 0] / model_config.out_h),
                x=float(detected_index[sample_i, detection_i, 1] / model_config.out_w),
                h=float(prediction.size[sample_i, detected_index[sample_i, detection_i, 0], detected_index[sample_i, detection_i, 1], 0]),
                w=float(prediction.size[sample_i, detected_index[sample_i, detection_i, 0], detected_index[sample_i, detection_i, 1], 1]),
                depth=float(depth[sample_i, detected_index[sample_i, detection_i, 0], detected_index[sample_i, detection_i, 1]]) if prediction.depth is not None else None,
                keypoints=[None] * n_keypoints,
                keypoint_scores=[None] * n_keypoints,
                keypoint_affinities=[None] * n_keypoints,
                cam_t_object=None,
            )

            sample_detections.append(detection)

        for keypoint_i in range(keypoint_n_detections):
            keypoint_score = float(detected_keypoint_score[sample_i, keypoint_i])
            if keypoint_score < keypoint_score_threshold:
                break

            keypoint_label = int(detected_keypoint_label[sample_i, keypoint_i])

            object_index, object_keypoint_index = object_config.decode_keypoint_index(keypoint_label)

            candidate_detections = [
                detection for detection in sample_detections
                if detection.label == object_index and detection.keypoints[object_keypoint_index] is None
            ]

            if len(candidate_detections) == 0:
                continue

            keypoint_y_index = detected_keypoint_index[sample_i, keypoint_i, 0]
            keypoint_x_index = detected_keypoint_index[sample_i, keypoint_i, 1]
            keypoint_y = float(keypoint_y_index / model_config.out_h)
            keypoint_x = float(keypoint_x_index / model_config.out_w)
            keypoint_affinity_y = float(prediction.keypoint_affinity[sample_i, keypoint_label, 0, keypoint_y_index, keypoint_x_index])
            keypoint_affinity_x = float(prediction.keypoint_affinity[sample_i, keypoint_label, 1, keypoint_y_index, keypoint_x_index])

            keypoint_affinity_angle = atan2(keypoint_affinity_y, keypoint_affinity_x)

            candidate_detection_angle_errors = [
                abs(keypoint_affinity_angle - atan2(keypoint_y - detection.y, keypoint_x - detection.x))
                for detection in candidate_detections
            ]

            match_detection = candidate_detections[candidate_detection_angle_errors.index(min(candidate_detection_angle_errors))]

            match_detection.keypoints[object_keypoint_index] = (keypoint_y, keypoint_x)
            match_detection.keypoint_affinities[object_keypoint_index] = (keypoint_affinity_y, keypoint_affinity_x)
            match_detection.keypoint_scores[object_keypoint_index] = keypoint_score

        for detection in sample_detections:
            # TODO: Implement partial keypoints
            n_keypoints = sum([1 if keypoint is not None else 0 for keypoint in detection.keypoints])

            if n_keypoints < 6:
                continue

            object_label = detection.label

            # keypoints_img = np.flip(np.concatenate((np.array([[detection.y, detection.x]]), np.array(detection.keypoints)), axis=0), axis=1)
            # keypoints_img_px = keypoints_img * np.array([model_config.in_w, model_config.in_h])
            # keypoints_cam = np.concatenate((np.array([[0, 0, 0]]), np.array(object_config.configs[object_label].keypoints)), axis=0)

            keypoints_img = []
            keypoints_cam = []

            for keypoint_i, keypoint in enumerate(detection.keypoints):
                if keypoint is not None:
                    keypoints_img.append([keypoint[1] * model_config.in_w, keypoint[0] * model_config.in_h])
                    keypoints_cam.append(object_config.configs[object_label].keypoints[keypoint_i])

            # keypoints_img = np.flip(np.array(detection.keypoints), axis=1)
            # keypoints_img_px = keypoints_img * np.array([model_config.in_w, model_config.in_h])
            # keypoints_cam = np.array(object_config.configs[object_label].keypoints)
            keypoints_img = np.array(keypoints_img)
            keypoints_cam = np.array(keypoints_cam)

            # print(keypoints_img_px)

            # success, rvec, tvec = cv2.solvePnP(keypoints_cam, keypoints_img, M_projection, None, cv2.SOLVEPNP_IPPE)
            success, rvec, tvec = cv2.solvePnP(keypoints_cam, keypoints_img, M_projection, None, cv2.SOLVEPNP_ITERATIVE)

            if success:
                rotm, _ = cv2.Rodrigues(rvec)

                match_detection.cam_t_object = SE3.Rt(SO3(rotm), tvec)

        detections.append(sample_detections)

    return detections


def decode(prediction: Prediction, model_config: ModelConfig,
           n_detections: int, score_threshold: float) -> [[Detection]]:

    heatmap = F.sigmoid(prediction.heatmap)
    heatmap = heatmap_nms(heatmap, kernel_size=3)
    detected_index, detected_label, detected_score = heatmap_detect(heatmap, n_detections)

    batch_size = detected_index.shape[0]

    detections = []

    if prediction.depth is not None:
        depth = depth_decode(prediction.depth)

    # if prediction.roll_bin is not None:
    #     roll = angle_decode(prediction.roll_bin, prediction.roll_offset, 2 * pi, pi / 3)
    #
    # if prediction.pitch_bin is not None:
    #     pitch = angle_decode(prediction.pitch_bin, prediction.pitch_offset, 2 * pi, pi / 3)
    #
    # if prediction.yaw_bin is not None:
    #     yaw = angle_decode(prediction.yaw_bin, prediction.yaw_offset, 2 * pi, pi / 3)

    # TODO: Decode angles

    for sample_i in range(batch_size):
        sample_detections = []

        for detection_i in range(n_detections):
            if detected_score[sample_i, detection_i] < score_threshold:
                break

            detection = Detection(
                label=detected_label[sample_i, detection_i],
                score=detected_score[sample_i, detection_i],
                y=(model_config.downsample_ratio * float(detected_index[sample_i, detection_i, 0]) + float(prediction.offset[sample_i, detected_index[sample_i, detection_i, 0], detected_index[sample_i, detection_i, 1], 0])) / model_config.in_h,
                x=(model_config.downsample_ratio * float(detected_index[sample_i, detection_i, 1]) + float(prediction.offset[sample_i, detected_index[sample_i, detection_i, 0], detected_index[sample_i, detection_i, 1], 1])) / model_config.in_w,
                h=float(prediction.size[sample_i, detected_index[sample_i, detection_i, 0], detected_index[sample_i, detection_i, 1], 0]),
                w=float(prediction.size[sample_i, detected_index[sample_i, detection_i, 0], detected_index[sample_i, detection_i, 1], 1]),
            )

            if prediction.depth is not None:
                detection.depth = float(depth[sample_i, detected_index[sample_i, detection_i, 0], detected_index[sample_i, detection_i, 1], 0])
            #
            # if prediction.roll_bin is not None:
            #     prediction.roll = float(roll[sample_i, detected_index[sample_i, detection_i, 0], detected_index[sample_i, detection_i, 1]])
            #
            # if prediction.pitch_bin is not None:
            #     prediction.pitch = float(pitch[sample_i, detected_index[sample_i, detection_i, 0], detected_index[sample_i, detection_i, 1]])
            #
            # if prediction.yaw_bin is not None:
            #     prediction.yaw = float(yaw[sample_i, detected_index[sample_i, detection_i, 0], detected_index[sample_i, detection_i, 1]])

            sample_detections.append(detection)

        detections.append(sample_detections)

    return detections


def heatmap_nms(heatmap: torch.Tensor, kernel_size: int) -> torch.Tensor:
    # heatmap is [batch_size, n_heatmaps, feature_h, feature_w]
    # result is [batch_size, n_heatmaps, feature_h, feature_w]

    assert kernel_size >= 1 and kernel_size % 2 == 1

    heatmap_max = F.max_pool2d(
        heatmap,
        (kernel_size, kernel_size),
        stride=1,
        padding=(kernel_size - 1) // 2,
    )

    return (heatmap_max == heatmap).float() * heatmap


def heatmap_detect(heatmap: torch.Tensor, n_detections: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # heatmap is [batch_size, n_heatmaps, feature_h, feature_w]
    #
    # result is (detected_index, detected_label, detected_score)
    # index is [batch_size, n_detections, 2]
    #   index[:, :, 0] is y index in [0, feature_h)
    #   index[:, :, 1] is x index in [0, feature_w)
    # label is [batch_size, n_detections]
    # score is [batch_size, n_detections]

    batch_size, n_heatmaps, feature_h, feature_w = heatmap.shape

    scores = heatmap.reshape(batch_size, -1)

    selected_score, selected_index = torch.topk(scores, n_detections)

    selected_label = (selected_index / (feature_h * feature_w)).to(torch.long)
    selected_index = (selected_index % (feature_h * feature_w)).to(torch.long)

    selected_index = torch.stack((
        (selected_index / feature_w).to(torch.long),
        (selected_index % feature_w).to(torch.long),
    ), dim=-1)

    return selected_index, selected_label, selected_score


def angle_get_bins(bin_overlap: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    # bins are ((bin_0_center, bin_0_min, bin_0_max), (bin_1_center, bin_1_min, bin_1_max))

    bin_0 = (pi / 2, -bin_overlap / 2, pi + bin_overlap / 2)
    bin_1 = (-pi / 2, -pi - bin_overlap / 2, bin_overlap / 2)

    return bin_0, bin_1


def angle_decode(predicted_bin: torch.Tensor, predicted_offset: torch.Tensor, theta_range: float, bin_overlap: float) -> torch.Tensor:
    # predicted_bin is [batch_size, n_detections, 4]
    #   predicted_bin[0:2] are [outside, inside] classifications for bin 0
    #   predicted_bin[2:4] are [outside, inside] classifications for bin 1
    # predicted_offset is [batch_size, n_detections, 4]
    #   predicted_offset[0:2] are [sin, cos] offsets for bin 0
    #   predicted_offset[2:4] are [sin, cos] offsets for bin 1
    #
    # result is [batch_size, n_detections]

    bins = angle_get_bins(bin_overlap)
    (bin_0_center, bin_0_min, bin_0_max), (bin_1_center, bin_1_min, bin_1_max) = bins

    classification_score_bin_0 = F.softmax(predicted_bin[:, :, 0:2], dim=-1)[:, :, 1]    # [batch_size, n_detections]
    classification_score_bin_1 = F.softmax(predicted_bin[:, :, 2:4], dim=-1)[:, :, 1]    # [batch_size, n_detections]

    use_bin_1 = classification_score_bin_1 > classification_score_bin_0    # [batch_size, n_detections]

    angle_bin_0 = bin_0_center + torch.atan2(predicted_offset[:, :, 0], predicted_offset[:, :, 1])    # [batch_size, n_detections]
    angle_bin_1 = bin_1_center + torch.atan2(predicted_offset[:, :, 2], predicted_offset[:, :, 3])    # [batch_size, n_detections]

    angle = torch.where(use_bin_1, angle_bin_1, angle_bin_0)   # [batch_size, n_detections]
    angle = angle % (2 * pi)
    angle = angle * (theta_range / (2 * pi))

    return angle


def depth_decode(prediction: torch.Tensor) -> torch.Tensor:
    # prediction is [size]
    #
    # result is [size]

    return (1 / F.sigmoid(prediction)) - 1


def main():
    from tauv_vision.centernet.model.loss import gaussian_splat

    heatmap = torch.cat((
        gaussian_splat(512, 512, 100, 100, 50).unsqueeze(0).unsqueeze(1),
        gaussian_splat(512, 512, 200, 200, 50).unsqueeze(0).unsqueeze(1),
    ), dim=1)

    heatmap = heatmap_nms(heatmap, 3)

    detected_index, detected_label, detected_score = heatmap_detect(heatmap, 100)

    assert detected_index[0, 0, 0] == 100 and detected_index[0, 0, 1] == 100


if __name__ == "__main__":
    main()
