import cv2
import torch
from torch.utils.data import DataLoader
from math import pi, sqrt
import pathlib
import albumentations as A
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

from tauv_vision.centernet.model.centernet import Centernet, initialize_weights
from tauv_vision.centernet.model.backbones.dla import DLABackbone
from tauv_vision.centernet.model.loss import loss
from tauv_vision.centernet.model.config import ObjectConfig, ObjectConfigSet, AngleConfig, ModelConfig, TrainConfig
from tauv_vision.datasets.load.pose_dataset import PoseDataset, PoseSample, Split
from tauv_vision.centernet.model.decode import decode_keypoints, KeypointDetection


model_config = ModelConfig(
    in_h=360,
    in_w=640,
    backbone_heights=[2, 2, 2, 2, 2, 2],
    backbone_channels=[128, 128, 128, 128, 128, 128, 128],
    downsamples=2,
    angle_bin_overlap=pi / 3,
)

object_config = ObjectConfigSet(
    configs=[
        ObjectConfig(
            id="torpedo_22_trapezoid",
            yaw=AngleConfig(
                train=False,
                modulo=2 * pi,
            ),
            pitch=AngleConfig(
                train=False,
                modulo=2 * pi,
            ),
            roll=AngleConfig(
                train=False,
                modulo=2 * pi,
            ),
            train_depth=False,
            train_keypoints=True,
            keypoints=[
                (0.0, 0.1, 0.1),
                (0.0, 0.1, -0.1),
                (0.0, -0.1, -0.08),
                (0.0, -0.1, 0.08),
            ],
        ),
    ]
)

test_dataset_root = pathlib.Path("~/Documents/TAUV-Datasets/pay-large-president").expanduser()

checkpoint = pathlib.Path("~/Documents/centernet_runs/2.pt").expanduser()


def is_positive_detection(detection, truth, distance_threshold) -> bool:

    if int(detection.label) != int(truth.label):
        return False

    distance = sqrt((detection.y - truth.y) ** 2 + (detection.x - truth.x) ** 2)

    if distance > distance_threshold:
        return False

    return True


def get_truth_detections(batch, model_config: ModelConfig) -> [[KeypointDetection]]:
    batch_size, n_detections = batch.valid.shape

    detections = []

    for sample_i in range(batch_size):
        sample_detections = []

        for detection_i in range(n_detections):
            if not batch.valid[sample_i, detection_i]:
                continue

            detection = KeypointDetection(
                label=batch.label[sample_i, detection_i],
                score=1,
                y=batch.center[sample_i, detection_i, 0],
                x=batch.center[sample_i, detection_i, 1],
                keypoints=None,
                keypoint_scores=None,
                keypoint_affinities=None,
                cam_t_object=None, # TODO: Probably fill this in?
            )

            sample_detections.append(detection)

        detections.append(sample_detections)

    return detections


def evaluate_precision_recall(centernet: Centernet, model_config: ModelConfig, object_config, M_projection, data_loader, device, score_threshold, keypoint_score_threshold, keypoint_angle_threshold, distance_threshold) -> Tuple[float, float]:
    n_tp = 0
    n_detections = 0
    n_truth_detections = 0

    for batch_i, batch in enumerate(data_loader):
        if batch_i > 100:
            break

        print(f"evaluating {batch_i}")
        batch = batch.to(device)

        # TODO: Add something to normalize img!

        prediction = centernet.forward(batch.img)

        detections = decode_keypoints(prediction, model_config, object_config, M_projection, 100, 100, score_threshold, keypoint_score_threshold, keypoint_angle_threshold)

        truth_detections = get_truth_detections(batch, model_config)

        for sample_i, (sample_detections, sample_truth_detections) in enumerate(zip(detections, truth_detections)):

            img_np = np.array(batch.img[sample_i].permute(1, 2, 0).flip(-1).detach().cpu()).copy()

            for detection in sample_detections:
                if detection.cam_t_object is None:
                    continue

                rvec, _ = cv2.Rodrigues(detection.cam_t_object.R)
                tvec = detection.cam_t_object.t

                cv2.drawFrameAxes(img_np, M_projection, None, rvec, tvec, 0.1, 3)

            plt.imshow(img_np)
            plt.show()

            n_detections += len(sample_detections)
            n_truth_detections += len(sample_truth_detections)

            for detection in reversed(sorted(sample_detections, key=lambda d: d.score)):

                if len(sample_truth_detections) == 0:
                    continue


                for truth_detection in sample_truth_detections:
                    if is_positive_detection(detection, truth_detection, distance_threshold):
                        n_tp += 1
                        sample_truth_detections.remove(truth_detection)
                        break

    precision = n_tp / n_detections if n_detections > 0 else 1
    recall = n_tp / n_truth_detections

    return (precision, recall)


def evaluate_precision_recall_curve(centernet: Centernet, model_config, object_config, M_projection, data_loader, device, keypoint_score_threshold, keypoint_angle_threshold, distance_threshold):

    score_thresholds = torch.linspace(0.9, 1, 10)

    precision = torch.zeros(score_thresholds.shape[0])
    recall = torch.zeros(score_thresholds.shape[0])

    for i, score_threshold in enumerate(score_thresholds):
        (p, r) = evaluate_precision_recall(centernet, model_config, object_config, M_projection, data_loader, device, score_threshold, keypoint_score_threshold, keypoint_angle_threshold, distance_threshold)

        print(score_threshold, p, r)

        precision[i] = p
        recall[i] = r

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.show()


def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"running on {device}")

    test_transform = A.Compose(
        [
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.3, 0.3, 0.3), always_apply=True)
        ]
    )

    dla_backbone = DLABackbone(model_config.backbone_heights, model_config.backbone_channels, model_config.downsamples)
    centernet = Centernet(dla_backbone, object_config).to(device)

    centernet.load_state_dict(torch.load(checkpoint, map_location=device))

    centernet.eval()

    test_dataset = PoseDataset(test_dataset_root, Split.VAL, object_config.label_id_to_index, object_config, test_transform)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=PoseSample.collate,
    )

    M_projection = np.array([
        [732, 0, 320],
        [0, 732, 180],
        [0, 0, 1],
    ], dtype=np.float32)

    evaluate_precision_recall_curve(centernet, model_config, object_config, M_projection, test_dataloader, device, keypoint_score_threshold=0.5, keypoint_angle_threshold=0.5, distance_threshold=0.01)

    # for batch_i, batch in enumerate(test_dataloader):
    #     batch = batch.to(device)
    #
    #     prediction = centernet.forward(batch.img)
    #
    #     pass


if __name__ == "__main__":
    main()