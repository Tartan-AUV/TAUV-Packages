import numpy as np
from torch.utils.data import DataLoader
import torch
import cv2
import albumentations as A
import pathlib
from math import pi
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as T

from tauv_vision.centernet.model.centernet import Centernet, initialize_weights
from tauv_vision.centernet.model.backbones.dla import DLABackbone
from tauv_vision.centernet.model.loss import loss
from tauv_vision.centernet.model.config import ObjectConfig, ObjectConfigSet, AngleConfig, ModelConfig, TrainConfig
from tauv_vision.datasets.load.pose_dataset import PoseDataset, PoseSample, Split
from tauv_vision.centernet.model.decode import decode_keypoints, KeypointDetection
from tauv_vision.centernet.model.decode import decode, Detection

# model_config = ModelConfig(
#     in_h=360,
#     in_w=640,
#     backbone_heights=[2, 2, 2, 2, 2, 2],
#     backbone_channels=[128, 128, 128, 128, 128, 128, 128],
#     downsamples=1,
#     angle_bin_overlap=pi / 3,
# )
model_config = ModelConfig(
    in_h=360,
    in_w=640,
    backbone_heights=[2, 2, 2, 2, 2],
    backbone_channels=[128, 128, 128, 128, 128, 128],
    downsamples=1,
    angle_bin_overlap=pi / 3,
)

object_config = ObjectConfigSet(
    configs=[
        # ObjectConfig(
        #     id="torpedo_22_trapezoid",
        #     yaw=AngleConfig(
        #         train=False,
        #         modulo=2 * pi,
        #     ),
        #     pitch=AngleConfig(
        #         train=False,
        #         modulo=2 * pi,
        #     ),
        #     roll=AngleConfig(
        #         train=False,
        #         modulo=2 * pi,
        #     ),
        #     train_depth=False,
        #     train_keypoints=True,
        #     keypoints=[
        #         (0.0, 0.1, 0.1),
        #         (0.0, 0.1, -0.1),
        #         (0.0, -0.1, -0.08),
        #         (0.0, -0.1, 0.08),
        #         (0.0, 0.509, 0.432),
        #         (0.0, 0.223, 0.337),
        #         (0.0, 0.398, 0.207),
        #         (0.0, 0.334, 0.063),
        #         (0.0, -0.112, 0.278),
        #         (0.0, 0.269, -0.062),
        #     ],
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
            train_depth=False,
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
            train_depth=False,
            train_keypoints=True,
            keypoints=[
                (0, 0.15364, 0),
                (0, 0.01447, 0.06353),
                (0.04461, 0.01447, -0.07617),
                (-0.04461, 0.01447, -0.07617),
                (0.11042, 0.04946, -0.09637),
                (-0.11011, 0.04946, -0.05596),
                (0.07636, 0.04576, 0.04493),
                (0.07636, 0.13557, -0.04487),
                # (-0.07617, 0.01447, 0.04461),
                # (-0.07617, 0.01447, 0.04461),
                # (-0.0442, 0.01147, 0),
                # (-0.09637, 0.04946, 0.11042),
                # (-0.05596, 0.04946, 0.11011),
                # (0, 0.09067, 0.04461),
                # (-0.04487, 0.13557, 0.07636),
                # (0.04493, 0.04576, 0.07636)
            ]
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
            train_depth=False,
            train_keypoints=True,
            keypoints=[
                (0.04828, 0.09168, -0.08136),
                (0.09411, 0.01364, 0.14643),
                (-0.04681, 0.01364, 0.08806),
                (0.10495, 0.01364, -0.06372),
                (0.14247, 0.03766, 0.04158),
                (0.04005, 0.07867, 0.07143),
                (0.00363, 0.08621, -0.03654)
            ]
        ),
    ]
)

checkpoint = pathlib.Path("~/Documents/centernet_runs/69.pt").expanduser()

test_dataset_root = pathlib.Path("~/Documents/TAUV-Datasets/stop-fine-mother").expanduser()

def main():
    device = torch.device("cpu")

    dla_backbone = DLABackbone(model_config.backbone_heights, model_config.backbone_channels, model_config.downsamples)
    centernet = Centernet(dla_backbone, object_config).to(device)

    centernet.load_state_dict(torch.load(checkpoint, map_location=device))

    centernet.eval()

    test_transform = A.Compose(
        [
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.3, 0.3, 0.3), always_apply=True)
        ]
    )

    # transform = A.Compose(
    #     [
    #         A.Resize(360, 640),
    #         A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.3, 0.3, 0.3), always_apply=True)
    #     ]
    # )
    # to_tensor = T.Compose([T.ToImageTensor(), T.ConvertImageDtype()])

    test_dataset = PoseDataset(test_dataset_root, Split.VAL, object_config.label_id_to_index, object_config, test_transform)

    M_projection = np.array([
        [732, 0, 320],
        [0, 732, 180],
        [0, 0, 1],
    ], dtype=np.float32)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=PoseSample.collate,
    )

    for sample_i, sample in enumerate(test_dataset):
        with torch.no_grad():
            prediction = centernet.forward(sample.img)

        # detections = decode(
        #     prediction,
        #     model_config,
        #     n_detections=10,
        #     score_threshold=0.1,
        # )[0]
        detections = decode_keypoints(
            prediction,
            model_config,
            object_config,
            M_projection,
            n_detections=10,
            keypoint_n_detections=10,
            score_threshold=0.5,
            keypoint_score_threshold=0.2,
            keypoint_angle_threshold=0.5,
        )[0]

        print(len(detections))

        frame = sample.img[0].permute(1, 2, 0).detach().cpu().numpy()
        frame = 255 * (frame - frame.min()) / (frame.max() - frame.min())
        frame = frame.astype(np.uint8).copy()

        pass

        for detection in detections:
            cv2.circle(frame, (int(detection.x * 640), int(detection.y * 360)), 3, (255, 0, 0), -1)

            if detection.cam_t_object is not None:
                rvec, _ = cv2.Rodrigues(detection.cam_t_object.R)
                tvec = detection.cam_t_object.t

                cv2.drawFrameAxes(frame, M_projection, None, rvec, tvec, 0.1, 3)

        plt.imshow(frame)
        plt.show()


if __name__ == "__main__":
    main()