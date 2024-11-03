import torch
from torch.utils.data import DataLoader
from math import pi
import pathlib
import albumentations as A
from typing import Tuple
import matplotlib.pyplot as plt

from tauv_vision.centernet.model.centernet import Centernet, initialize_weights
from tauv_vision.centernet.model.backbones.dla import DLABackbone
from tauv_vision.centernet.model.loss import loss
from tauv_vision.centernet.model.config import ObjectConfig, ObjectConfigSet, AngleConfig, ModelConfig, TrainConfig
from tauv_vision.datasets.load.pose_dataset import PoseDataset, PoseSample, Split
from tauv_vision.centernet.model.decode import decode, Detection


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
        # ObjectConfig(
        #     id="torpedo_22_circle",
        #     yaw=AngleConfig(
        #         train=True,
        #         modulo=2 * pi,
        #     ),
        #     pitch=AngleConfig(
        #         train=True,
        #         modulo=2 * pi,
        #     ),
        #     roll=AngleConfig(
        #         train=True,
        #         modulo=2 * pi,
        #     ),
        #     train_depth=True,
        # ),
        ObjectConfig(
            id="sample_24_worm",
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
            train_depth=True,
            train_keypoints=False,
            keypoints=[]
        ),
        ObjectConfig(
            id="sample_24_coral",
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
            train_depth=True,
            train_keypoints=False,
            keypoints=[]
        ),
        ObjectConfig(
            id="sample_24_nautilus",
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
            train_depth=True,
            train_keypoints=False,
            keypoints=[]
        ),
    ]
)

test_dataset_root = pathlib.Path("~/Documents/TAUV-Datasets/stop-fine-mother").expanduser()

checkpoint = pathlib.Path("~/Documents/centernet_runs/16.pt").expanduser()


def iou(d1, d2) -> bool:
    ya = max(d1.y - d1.h / 2, d2.y - d2.h / 2)
    xa = max(d1.x - d1.w / 2, d2.x - d2.w / 2)
    yb = min(d1.y + d1.h / 2, d2.y + d2.h / 2)
    xb = min(d1.x + d1.w / 2, d2.x + d2.w / 2)

    intersection = abs(max(yb - ya, 0) * max(xb - xa, 0))

    if intersection == 0:
        return intersection

    a1 = d1.w * d1.h
    a2 = d2.w * d2.h

    result = intersection / (a1 + a2 - intersection)

    return result


def is_positive_detection(detection, truth, iou_threshold) -> bool:

    if int(detection.label) != int(truth.label):
        return False

    return iou(detection, truth) >= iou_threshold


def get_truth_detections(batch, model_config: ModelConfig) -> [[Detection]]:
    batch_size, n_detections = batch.valid.shape

    detections = []

    for sample_i in range(batch_size):
        sample_detections = []

        for detection_i in range(n_detections):
            if not batch.valid[sample_i, detection_i]:
                continue

            detection = Detection(
                label=batch.label[sample_i, detection_i],
                score=1,
                y=batch.center[sample_i, detection_i, 0],
                x=batch.center[sample_i, detection_i, 1],
                h=batch.size[sample_i, detection_i, 0],
                w=batch.size[sample_i, detection_i, 1],
            )

            detection.depth = batch.depth[sample_i, detection_i]

            detection.roll = batch.roll[sample_i, detection_i]
            detection.pitch = batch.pitch[sample_i, detection_i]
            detection.yaw = batch.yaw[sample_i, detection_i]

            sample_detections.append(detection)

        detections.append(sample_detections)

    return detections


def evaluate_precision_recall(centernet: Centernet, model_config: ModelConfig, data_loader, device, score_threshold, iou_threshold) -> Tuple[float, float]:
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

        detections = decode(prediction, model_config, 100, score_threshold)

        truth_detections = get_truth_detections(batch, model_config)


        for sample_i, (sample_detections, sample_truth_detections) in enumerate(zip(detections, truth_detections)):

            n_detections += len(sample_detections)
            n_truth_detections += len(sample_truth_detections)

            for detection in reversed(sorted(sample_detections, key=lambda d: d.score)):

                if len(sample_truth_detections) == 0:
                    continue


                for truth_detection in sample_truth_detections:
                    if is_positive_detection(detection, truth_detection, iou_threshold):
                        n_tp += 1
                        sample_truth_detections.remove(truth_detection)
                        break

    precision = n_tp / n_detections if n_detections > 0 else 1
    recall = n_tp / n_truth_detections

    return (precision, recall)


def evaluate_precision_recall_curve(centernet: Centernet, model_config, data_loader, device, iou_threshold):

    score_thresholds = torch.linspace(0, 1, 10)

    precision = torch.zeros(score_thresholds.shape[0])
    recall = torch.zeros(score_thresholds.shape[0])

    for i, score_threshold in enumerate(score_thresholds):
        (p, r) = evaluate_precision_recall(centernet, model_config, data_loader, device, score_threshold, iou_threshold)

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

    evaluate_precision_recall_curve(centernet, model_config, test_dataloader, device, iou_threshold=0.1)

    # for batch_i, batch in enumerate(test_dataloader):
    #     batch = batch.to(device)
    #
    #     prediction = centernet.forward(batch.img)
    #
    #     pass


if __name__ == "__main__":
    main()