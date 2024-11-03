from enum import Enum
import numpy as np
import pathlib
import random
import json
from torch.utils.data import Dataset
from dataclasses import dataclass
import torch
from typing import Optional, Dict
from typing_extensions import Self
from PIL import Image
import torchvision.transforms.v2 as T
import torch.nn.functional as F

from tauv_vision.centernet.model.config import ObjectConfigSet


class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class PoseSample:
    img: torch.Tensor # [batch_size, 3, in_h, in_w]

    valid: torch.Tensor  # [batch_size, n_objects]
    label: torch.Tensor  # [batch_size, n_objects]
    center: torch.Tensor  # [batch_size, n_objects, 2]
    size: torch.Tensor  # [batch_size, n_objects, 2]

    roll: Optional[torch.Tensor]  # [batch_size, n_objects]
    pitch: Optional[torch.Tensor]  # [batch_size, n_objects]
    yaw: Optional[torch.Tensor]  # [batch_size, n_objects]
    depth: Optional[torch.Tensor]  # [batch_size, n_objects]

    keypoint_valid: Optional[torch.Tensor]        # [batch_size, n_keypoint_instances]
    keypoint_label: Optional[torch.Tensor]        # [batch_size, n_keypoint_instances]
    keypoint_center: Optional[torch.Tensor]       # [batch_size, n_keypoint_instances, 2]
    keypoint_object_index: Optional[torch.Tensor] # [batch_size, n_keypoint_instances]

    def to(self, device: torch.device) -> Self:
        return PoseSample(
            img=self.img.to(device),
            valid=self.valid.to(device),
            label=self.label.to(device),
            center=self.center.to(device),
            size=self.size.to(device),
            roll=self.roll.to(device) if self.roll is not None else None,
            pitch=self.pitch.to(device) if self.pitch is not None else None,
            yaw=self.yaw.to(device) if self.yaw is not None else None,
            depth=self.depth.to(device) if self.depth is not None else None,
            keypoint_valid=self.keypoint_valid.to(device) if self.keypoint_valid is not None else None,
            keypoint_label=self.keypoint_label.to(device) if self.keypoint_label is not None else None,
            keypoint_center=self.keypoint_center.to(device) if self.keypoint_center is not None else None,
            keypoint_object_index=self.keypoint_object_index.to(device) if self.keypoint_object_index is not None else None,
        )

    @classmethod
    def load(cls, data_path: pathlib.Path, id: str, label_id_to_index: Dict[str, int], object_config: ObjectConfigSet, transform) -> Self:
        json_path = (data_path / id).with_suffix(".json")
        img_path = (data_path / id).with_suffix(".png")

        with open(json_path, "r") as fp:
            data = json.load(fp)

        img_pil = Image.open(img_path).convert("RGB")

        to_tensor = T.Compose([T.ToImageTensor(), T.ConvertImageDtype()])

        img_np = np.array(img_pil.convert("RGB"))

        filtered_objects = [
            object for object in data["objects"]
            if object["label"] in label_id_to_index # and object["bbox"]["h"] > 0.01 and object["bbox"]["w"] > 0.01
        ]

        n_objects = len(filtered_objects)

        filtered_object_configs = [object_config.get_by_label(object["label"]) for object in filtered_objects]

        n_keypoint_instances = sum([
            len(object.keypoints) if object.train_keypoints else 0 for object in filtered_object_configs
        ])

        M_projection = torch.Tensor(data["camera"]["projection"]).reshape(3, 4)

        bboxes_np = np.zeros((n_objects, 4))
        bbox_labels_np = np.zeros((n_objects,), dtype=int)
        bbox_indices_np = np.zeros((n_objects,), dtype=int)

        roll_np = np.zeros((n_objects,))
        pitch_np = np.zeros((n_objects,))
        yaw_np = np.zeros((n_objects,))
        depth_np = np.zeros((n_objects,))
        keypoints_np = np.zeros((n_keypoint_instances, 2))

        keypoint_labels_np = np.zeros((n_keypoint_instances,), dtype=int)
        keypoint_object_indices_np = np.zeros((n_keypoint_instances,), dtype=int)

        keypoint_instance_i = 0

        for i, object in enumerate(filtered_objects):
            object_index = label_id_to_index[object["label"]]

            xmin = object["bbox"]["x"] - object["bbox"]["w"] / 2
            xmax = object["bbox"]["x"] + object["bbox"]["w"] / 2
            ymin = object["bbox"]["y"] - object["bbox"]["h"] / 2
            ymax = object["bbox"]["y"] + object["bbox"]["h"] / 2

            bboxes_np[i] = np.clip(np.array([xmin, ymin, xmax, ymax]), 0, 1)

            if bboxes_np[i, 0] == bboxes_np[i, 2]:
                bboxes_np[i, 2] += 0.01
            if bboxes_np[i, 1] == bboxes_np[i, 3]:
                bboxes_np[i, 3] += 0.01

            bboxes_np[i] = np.clip(bboxes_np[i], 0, 1)
            bbox_indices_np[i] = i
            bbox_labels_np[i] = object_index

            roll_np[i] = object["pose"]["roll"]
            pitch_np[i] = object["pose"]["pitch"]
            yaw_np[i] = object["pose"]["yaw"]
            depth_np[i] = object["pose"]["distance"]

            M_cam_t_object = torch.Tensor(object["pose"]["cam_t_object"]).reshape(4, 4)

            config = filtered_object_configs[i]

            if config.keypoints is not None:
                for object_keypoint_index, object_keypoint in enumerate(config.keypoints):
                    keypoint_object_h = torch.Tensor([object_keypoint[0], object_keypoint[1], object_keypoint[2], 1])
                    keypoint_cam_h = torch.matmul(M_cam_t_object, keypoint_object_h)
                    keypoint_cam_2d_h = torch.matmul(M_projection, keypoint_cam_h)
                    keypoint_cam_2d = keypoint_cam_2d_h[:2] / keypoint_cam_2d_h[2]

                    if 0 <= keypoint_cam_2d[0] < data["camera"]["w"] \
                            and 0 <= keypoint_cam_2d[1] < data["camera"]["h"]:
                        keypoint_labels_np[keypoint_instance_i] = object_config.encode_keypoint_index(object_index, object_keypoint_index)
                        keypoint_object_indices_np[keypoint_instance_i] = i

                        keypoints_np[keypoint_instance_i, 0] = keypoint_cam_2d[0]
                        keypoints_np[keypoint_instance_i, 1] = keypoint_cam_2d[1]

                        keypoint_instance_i += 1

        n_keypoint_instances = keypoint_instance_i
        keypoints_np = keypoints_np[:n_keypoint_instances]
        keypoint_labels_np = keypoint_labels_np[:n_keypoint_instances]
        keypoint_object_indices_np = keypoint_object_indices_np[:n_keypoint_instances]

        if transform is not None:
            transformed = transform(
                image=img_np,
                bboxes=bboxes_np,
                bbox_labels=bbox_labels_np,
                bbox_indices=bbox_indices_np,
                roll=roll_np,
                pitch=pitch_np,
                yaw=yaw_np,
                depth=depth_np,
                keypoints=keypoints_np,
                keypoint_labels=keypoint_labels_np,
                keypoint_object_indices=keypoint_object_indices_np,
            )

            img_np = transformed["image"]
            bboxes_np = np.array(transformed["bboxes"]) if len(transformed["bboxes"]) > 0 else np.zeros((0, 4))
            bbox_labels_np = np.array(transformed["bbox_labels"], dtype=int)
            bbox_indices_np = np.array(transformed["bbox_indices"], dtype=int)
            roll_np = np.array(transformed["roll"])
            pitch_np = np.array(transformed["pitch"])
            yaw_np = np.array(transformed["yaw"])
            depth_np = np.array(transformed["depth"])
            keypoints_np = np.array(transformed["keypoints"]) if len(transformed["keypoints"]) > 0 else np.zeros((0, 2))
            keypoint_labels_np = np.array(transformed["keypoint_labels"], dtype=int)
            keypoint_object_indices_np = np.array(transformed["keypoint_object_indices"], dtype=int)

        img = to_tensor(img_np)

        n_objects = bboxes_np.shape[0]
        n_keypoint_instances = keypoints_np.shape[0]

        valid = torch.full((n_objects,), fill_value=True, dtype=torch.bool)

        label = torch.Tensor(bbox_labels_np).to(torch.long)

        center = torch.Tensor(np.stack((
            (bboxes_np[:, 1] + bboxes_np[:, 3]) / 2,
            (bboxes_np[:, 0] + bboxes_np[:, 2]) / 2,
        ), axis=-1))

        size = torch.Tensor(np.stack((
            (bboxes_np[:, 3] - bboxes_np[:, 1]),
            (bboxes_np[:, 2] - bboxes_np[:, 0]),
        ), axis=-1))

        roll = torch.Tensor(roll_np)
        pitch = torch.Tensor(pitch_np)
        yaw = torch.Tensor(yaw_np)
        depth = torch.Tensor(depth_np)

        keypoint_valid = torch.full((n_keypoint_instances,), fill_value=True, dtype=torch.bool)
        keypoint_center = torch.Tensor(np.stack((
            keypoints_np[:, 1] / data["camera"]["h"],
            keypoints_np[:, 0] / data["camera"]["w"],
        ), axis=-1))
        keypoint_label = torch.Tensor(keypoint_labels_np).to(torch.long)

        for keypoint_i, keypoint_object_index in enumerate(keypoint_object_indices_np):
            for bbox_i, bbox_index in enumerate(bbox_indices_np):
                if bbox_index == keypoint_object_index:
                    keypoint_object_indices_np[keypoint_i] = bbox_i
                    break

        keypoint_object_index = torch.Tensor(keypoint_object_indices_np).to(torch.long)

        # keypoint_label = torch.zeros(n_keypoint_instances, dtype=torch.long)
        # keypoint_center = torch.zeros((n_keypoint_instances, 2), dtype=torch.float32)
        # keypoint_object_index = torch.zeros(n_keypoint_instances, dtype=torch.long)

        # keypoint_instance_i = 0

        # for i, object in enumerate(filtered_objects):
        #     object_index = label_id_to_index[object["label"]]
        #     label[i] = object_index
        #
        #     center[i, 0] = object["bbox"]["y"]
        #     center[i, 1] = object["bbox"]["x"]
        #
        #     size[i, 0] = object["bbox"]["h"]
        #     size[i, 1] = object["bbox"]["w"]
        #
        #     roll[i] = object["pose"]["roll"]
        #     pitch[i] = object["pose"]["pitch"]
        #     yaw[i] = object["pose"]["yaw"]
        #     depth[i] = object["pose"]["distance"]
        #
        #     M_cam_t_object = torch.Tensor(object["pose"]["cam_t_object"]).reshape(4, 4)
        #
        #     config = filtered_object_configs[i]
        #
        #     if config.keypoints is not None:
        #         for object_keypoint_index, object_keypoint in enumerate(config.keypoints):
        #             keypoint_label[keypoint_instance_i] = object_config.encode_keypoint_index(object_index, object_keypoint_index)
        #             keypoint_object_index[keypoint_instance_i] = i
        #
        #             keypoint_object_h = torch.Tensor([object_keypoint[0], object_keypoint[1], object_keypoint[2], 1])
        #             keypoint_cam_h = torch.matmul(M_cam_t_object, keypoint_object_h)
        #             keypoint_cam_2d_h = torch.matmul(M_projection, keypoint_cam_h)
        #             keypoint_cam_2d = keypoint_cam_2d_h[:2] / keypoint_cam_2d_h[2]
        #
        #             keypoint_center[keypoint_instance_i, 0] = keypoint_cam_2d[1] / data["camera"]["h"]
        #             keypoint_center[keypoint_instance_i, 1] = keypoint_cam_2d[0] / data["camera"]["w"]
        #
        #             keypoint_instance_i += 1

        sample = PoseSample(
            img=img.unsqueeze(0),
            valid=valid.unsqueeze(0),
            label=label.unsqueeze(0),
            center=center.unsqueeze(0),
            size=size.unsqueeze(0),
            roll=roll.unsqueeze(0),
            pitch=pitch.unsqueeze(0),
            yaw=yaw.unsqueeze(0),
            depth=depth.unsqueeze(0),
            keypoint_valid=keypoint_valid.unsqueeze(0),
            keypoint_label=keypoint_label.unsqueeze(0),
            keypoint_center=keypoint_center.unsqueeze(0),
            keypoint_object_index=keypoint_object_index.unsqueeze(0),
        )

        return sample

    @classmethod
    def collate(cls, samples: [Self]) -> Self:
        n_detections = [sample.valid.size(1) for sample in samples]
        max_n_detections = max(n_detections)

        n_keypoint_instances = [sample.keypoint_valid.size(1) for sample in samples]
        max_n_keypoint_instances = max(n_keypoint_instances)

        img = torch.cat([sample.img for sample in samples], dim=0)

        valid = torch.cat([
            F.pad(sample.valid, (0, max_n_detections - sample.valid.size(1)), value=False)
            for sample in samples
        ], dim=0)
        label = torch.cat([
            F.pad(sample.label, (0, max_n_detections - sample.label.size(1)), value=0)
            for sample in samples
        ], dim=0)
        center = torch.cat([
            F.pad(sample.center, (0, 0, 0, max_n_detections - sample.center.size(1)), value=0)
            for sample in samples
        ], dim=0)
        size = torch.cat([
            F.pad(sample.size, (0, 0, 0, max_n_detections - sample.size.size(1)), value=0)
            for sample in samples
        ], dim=0)
        roll = torch.cat([
            F.pad(sample.roll, (0, max_n_detections - sample.roll.size(1)), value=0)
            for sample in samples
        ], dim=0)
        pitch = torch.cat([
            F.pad(sample.pitch, (0, max_n_detections - sample.pitch.size(1)), value=0)
            for sample in samples
        ], dim=0)
        yaw = torch.cat([
            F.pad(sample.yaw, (0, max_n_detections - sample.yaw.size(1)), value=0)
            for sample in samples
        ], dim=0)
        depth = torch.cat([
            F.pad(sample.depth, (0, max_n_detections - sample.depth.size(1)), value=0)
            for sample in samples
        ], dim=0)

        keypoint_valid = torch.cat([
            F.pad(sample.keypoint_valid, (0, max_n_keypoint_instances - sample.keypoint_valid.size(1)), value=0)
            for sample in samples
        ], dim=0)
        keypoint_object_index = torch.cat([
            F.pad(sample.keypoint_object_index, (0, max_n_keypoint_instances - sample.keypoint_object_index.size(1)), value=0)
            for sample in samples
        ], dim=0)
        keypoint_label = torch.cat([
            F.pad(sample.keypoint_label, (0, max_n_keypoint_instances - sample.keypoint_label.size(1)), value=0)
            for sample in samples
        ], dim=0)
        keypoint_center = torch.cat([
            F.pad(sample.keypoint_center, (0, 0, 0, max_n_keypoint_instances - sample.keypoint_center.size(1)), value=0)
            for sample in samples
        ], dim=0)

        result = PoseSample(
            img=img,
            valid=valid,
            label=label,
            center=center,
            size=size,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            depth=depth,
            keypoint_valid=keypoint_valid,
            keypoint_object_index=keypoint_object_index,
            keypoint_label=keypoint_label,
            keypoint_center=keypoint_center,
        )

        return result


class PoseDataset(Dataset):

    def __init__(self, root: pathlib.Path, split: Split, label_id_to_index: Dict[str, int], object_config: ObjectConfigSet, transform):
        super().__init__()

        self._root_path: pathlib.Path = root
        self._split: Split = split

        if not self._root_path.is_dir():
            raise ValueError(f"No such directory: {self._root_path}")

        self._data_path: pathlib.Path = self._root_path / "data"

        if not self._data_path.is_dir():
            raise ValueError(f"No such directory: {self._data_path}")

        self._ids: [str] = self._get_ids()
        random.shuffle(self._ids)

        self._label_id_to_index: Dict[str, int] = label_id_to_index

        self._object_config = object_config

        self._transform = transform

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, i: int):
        return PoseSample.load(self._data_path, self._ids[i], self._label_id_to_index, self._object_config, self._transform)

    def _get_ids(self) -> [str]:
        splits_json_path = self._root_path / "splits.json"

        with open(splits_json_path, "r") as fp:
            splits_data = json.load(fp)

        return splits_data["splits"][self._split.value]


def main():
    label_id_to_index = {
        "torpedo_22_circle": 0,
        "torpedo_22_trapezoid": 1,
    }

    ds = PoseDataset(
        root=pathlib.Path("~/Documents/TAUV-Datasets/test").expanduser(),
        split=Split.TEST,
        label_id_to_index=label_id_to_index,
    )

    print(f"{len(ds)}")

    for sample in ds:
        print(sample)


if __name__ == "__main__":
    main()