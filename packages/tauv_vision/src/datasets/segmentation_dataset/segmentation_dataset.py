import torch
from torch.utils.data import Dataset
import pathlib
import numpy as np
import torchvision.transforms.v2 as T
import json
from PIL import Image
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from tauv_vision.yolact.model.boxes import box_xy_swap
import random


class SegmentationDatasetSet(Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


@dataclass
class SegmentationSample:
    img: torch.Tensor
    seg: torch.Tensor
    valid: torch.Tensor
    classifications: torch.Tensor
    bounding_boxes: torch.Tensor
    img_valid: torch.Tensor

    @classmethod
    def load(cls, data_path: pathlib.Path, class_ids_to_indices, id: str, transform: Optional = None):
        json_path = (data_path / id).with_suffix(".json")
        img_path = (data_path / id).with_suffix(".png")
        seg_path = (data_path / f"{id}_seg").with_suffix(".png")

        with open(json_path, "r") as fp:
            data = json.load(fp)

        img_pil = Image.open(img_path).convert("RGB")
        seg_pil = Image.open(seg_path)

        img_np = np.array(img_pil)
        seg_np = np.array(seg_pil)

        n_objects = len(data["objects"])

        valid = torch.full((n_objects,), fill_value=True, dtype=torch.bool)
        classifications_np = np.zeros(n_objects, dtype=np.int64)
        bounding_boxes_np = np.zeros((n_objects, 4))

        for i, object in enumerate(data["objects"]):
            classifications_np[i] = class_ids_to_indices[object["class_id"]]

            object_bbox = np.array([
                object["bbox"]["x"],
                object["bbox"]["y"],
                object["bbox"]["w"],
                object["bbox"]["h"],
            ])

            object_bbox_corners = np.array([
                object_bbox[0] - object_bbox[2] / 2,
                object_bbox[1] - object_bbox[3] / 2,
                object_bbox[0] + object_bbox[2] / 2,
                object_bbox[1] + object_bbox[3] / 2,
            ])

            object_bbox_corners = np.clip(object_bbox_corners, 0, 1)

            object_bbox = np.array([
                (object_bbox_corners[0] + object_bbox_corners[2]) / 2,
                (object_bbox_corners[1] + object_bbox_corners[3]) / 2,
                object_bbox_corners[2] - object_bbox_corners[0],
                object_bbox_corners[3] - object_bbox_corners[1],
            ])

            object_bbox = np.clip(object_bbox, 1e-3, 1-1e-3)

            bounding_boxes_np[i] = object_bbox

        if transform is not None:
            transformed = transform(
                image=img_np,
                mask=seg_np,
                bboxes=bounding_boxes_np,
                classifications=classifications_np,
            )

            img_np = transformed["image"]
            seg_np = transformed["mask"]
            bounding_boxes_np = transformed["bboxes"]
            classifications_np = transformed["classifications"]

        n_detections = len(bounding_boxes_np)

        img = T.ToTensor()(img_np)
        seg = T.ToTensor()(seg_np)[0]
        seg = (255 * seg).to(torch.uint8)
        img_valid = seg != 254
        classifications = torch.Tensor(classifications_np).to(torch.long)

        if n_detections == 0:
            valid = torch.Tensor([False])
            classifications = torch.Tensor([0]).to(torch.long)
            bounding_boxes = torch.Tensor([[0, 0, 0, 0]])

            sample = cls(
                img=img,
                seg=seg,
                valid=valid,
                classifications=classifications,
                bounding_boxes=bounding_boxes,
                img_valid=img_valid,
            )

            return sample

        bounding_boxes = box_xy_swap(torch.Tensor(bounding_boxes_np).unsqueeze(0)).squeeze(0)

        sample = cls(
            img=img,
            seg=seg,
            valid=valid,
            classifications=classifications,
            bounding_boxes=bounding_boxes,
            img_valid=img_valid
        )

        return sample


class SegmentationDataset(Dataset):

    def __init__(self, root: pathlib.Path, set: SegmentationDatasetSet, class_ids_to_indices, transform: Optional = None):
        super().__init__()

        self._class_ids_to_indices = class_ids_to_indices

        self._root_path: pathlib.Path = root
        self._set: SegmentationDatasetSet = set

        self._transform: Optional = transform

        if not self._root_path.is_dir():
            raise ValueError(f"No such directory: {self._root_path}")

        self._data_path: pathlib.Path = self._root_path / "data"

        if not self._data_path.is_dir():
            raise ValueError(f"No such directory: {self._data_path}")

        self._ids: [str] = self._get_ids()
        random.shuffle(self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, i: int) -> SegmentationSample:
        id = self._ids[i]

        return SegmentationSample.load(self._data_path, self._class_ids_to_indices, id, transform=self._transform)

    def _get_ids(self) -> [str]:
        splits_json_path = self._root_path / "splits.json"

        with open(splits_json_path, "r") as fp:
            splits_data = json.load(fp)

        return splits_data["splits"][self._set.value]


def main():
    ds = SegmentationDataset(
        root=pathlib.Path("~/Documents/torpedo_target_2").expanduser(),
        set=SegmentationDatasetSet.TRAIN,
    )

    print(f"Length: {len(ds)}")
    for sample in ds:
        print(sample)


if __name__ == "__main__":
    main()