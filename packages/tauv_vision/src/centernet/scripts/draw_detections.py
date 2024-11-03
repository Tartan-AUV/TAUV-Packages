import numpy as np
import torch
import cv2
import albumentations as A
import pathlib
from math import pi
import torchvision.transforms.v2 as T

from tauv_vision.centernet.model.backbones.centerpoint_dla import CenterpointDLA34
from tauv_vision.centernet.model.centernet import Centernet, initialize_weights
from tauv_vision.centernet.model.backbones.dla import DLABackbone
from tauv_vision.centernet.model.loss import loss
from tauv_vision.centernet.model.config import ObjectConfig, ObjectConfigSet, AngleConfig, ModelConfig, TrainConfig
from tauv_vision.datasets.load.pose_dataset import PoseDataset, PoseSample, Split
from tauv_vision.centernet.model.decode import decode_keypoints, KeypointDetection


model_config = ModelConfig(
    in_h=360,
    in_w=640,
    backbone_heights=[2, 2, 2, 2, 2],
    backbone_channels=[128, 128, 128, 128, 128, 128],
    downsamples=2,
    angle_bin_overlap=pi / 3,
)

train_config = TrainConfig(
    lr=5e-4,
    heatmap_focal_loss_a=2,
    heatmap_focal_loss_b=4,
    heatmap_sigma_factor=0.1,
    batch_size=32,
    n_batches=0,
    n_epochs=100,
    loss_lambda_keypoint_heatmap=1.0,
    loss_lambda_keypoint_affinity=0.01,
    keypoint_heatmap_sigma=2,
    keypoint_affinity_sigma=2,
    loss_lambda_size=0.1,
    loss_lambda_offset=0.0,
    loss_lambda_angle=0.0,
    loss_lambda_depth=0.1,
    n_workers=4,
    weight_save_interval=10,
)


object_config = ObjectConfigSet(
    configs=[
        ObjectConfig(
            id="sample_24_coral",
            yaw=AngleConfig(train=False, modulo=2 * pi),
            pitch=AngleConfig(train=False, modulo=2 * pi),
            roll=AngleConfig(train=False, modulo=2 * pi),
            train_depth=False,
            train_keypoints=True,
            keypoints=[
                (0, 0, 0)
            ]
        ),
        ObjectConfig(
            id="sample_24_nautilus",
            yaw=AngleConfig(train=False, modulo=2 * pi),
            pitch=AngleConfig(train=False, modulo=2 * pi),
            roll=AngleConfig(train=False, modulo=2 * pi),
            train_depth=False,
            train_keypoints=True,
            keypoints=[
                (0, 0, 0)
            ]
        ),
        ObjectConfig(
            id="torpedo_24",
            yaw=AngleConfig(train=False, modulo=2 * pi),
            pitch=AngleConfig(train=False, modulo=2 * pi),
            roll=AngleConfig(train=False, modulo=2 * pi),
            train_depth=False,
            train_keypoints=True,
            keypoints=[
                (0, 0, 0),
            ],
        ),
        ObjectConfig(
            id="torpedo_24_octagon",
            yaw=AngleConfig(train=False, modulo=2 * pi),
            pitch=AngleConfig(train=False, modulo=2 * pi),
            roll=AngleConfig(train=False, modulo=2 * pi),
            train_depth=False,
            train_keypoints=True,
            keypoints=[
                (0, 0, 0),
            ],
        ),
    ]
)
checkpoint = pathlib.Path("~/Documents/centernet_checkpoints/polished-salad-301_latest.pt").expanduser()

in_video = pathlib.Path("~/Downloads/oakd_front-color_2024-07-18-01-26-23_20.mp4").expanduser()
out_video = pathlib.Path("~/Downloads/oakd_front-color_2024-07-18-01-26-23_20.out.mp4").expanduser()

def main():
    device = torch.device("cuda")

    # dla_backbone = DLABackbone(model_config.backbone_heights, model_config.backbone_channels, model_config.downsamples)
    # centernet = Centernet(dla_backbone, object_config).to(device)

    centernet = CenterpointDLA34(object_config).to(device)

    centernet.load_state_dict(torch.load(checkpoint, map_location=device))

    centernet.eval()

    transform = A.Compose(
        [
            A.Resize(360, 640),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.3, 0.3, 0.3), always_apply=True)
        ]
    )
    to_tensor = T.Compose([T.ToImageTensor(), T.ConvertImageDtype()])

    M_projection = np.array([
        [732, 0, 320],
        [0, 732, 180],
        [0, 0, 1],
    ], dtype=np.float32)

    cap = cv2.VideoCapture(str(in_video))

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_video), fourcc, fps, (640, 360))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))

        img_np = transform(
            image=np.flip(frame, -1)
        )["image"]

        img = to_tensor(img_np).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = centernet.forward(img)

        detections = decode_keypoints(
            prediction,
            model_config,
            object_config,
            M_projection,
            n_detections=10,
            keypoint_n_detections=10,
            score_threshold=0.6,
            keypoint_score_threshold=0.1,
            keypoint_angle_threshold=0.1,
        )[0]

        print(len(detections))

        for detection in detections:
            cv2.circle(frame, (int(detection.x * 640), int(detection.y * 360)), 3, (255, 0, 0), -1)

            e_x = detection.x * 640
            e_y = detection.y * 360
            w = detection.w * 640
            h = detection.h * 360

            cv2.rectangle(
                frame,
                (int(e_x - 0.4 * w), int(e_y - 0.4 * h)),
                (int(e_x + 0.4 * w), int(e_y + 0.4 * h)),
                (0, 0, 255),
                1
            )

            cv2.putText(frame, f"{detection.score:02f}", (int(e_x - 0.4 * w), int(e_y - 0.5 * h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

            if detection.cam_t_object is None:
                continue

            print("frame")

            rvec, _ = cv2.Rodrigues(detection.cam_t_object.R)
            tvec = detection.cam_t_object.t

            cv2.drawFrameAxes(frame, M_projection, None, rvec, tvec, 0.1, 3)

        out.write(frame)

    cap.release()
    out.release()


if __name__ == "__main__":
    main()