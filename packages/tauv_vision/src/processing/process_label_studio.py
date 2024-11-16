import torch
import torchvision.transforms as T
from typing import Dict
from PIL import Image
import re
import glob
import argparse
import pathlib
import json
import numpy as np

from tauv_vision.datasets.segmentation_dataset.segmentation_dataset import SegmentationSample


"""FROM https://github.com/HumanSignal/label-studio-converter/blob/master/label_studio_converter/brush.py"""
class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i : self.i + size]
        self.i += size
        return int(out, 2)


def access_bit(data, num):
    """from bytes array to bits by num position"""
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def bytes2bit(data):
    """get bit string from bytes data"""
    return ''.join([str(access_bit(data, i)) for i in range(len(data) * 8)])


def decode_rle(rle, print_params: bool = False):
    """from LS RLE to numpy uint8 3d image [width, height, channel]

    Args:
        print_params (bool, optional): If true, a RLE parameters print statement is suppressed
    """
    input = InputStream(bytes2bit(rle))
    num = input.read(32)
    word_size = input.read(5) + 1
    rle_sizes = [input.read(4) + 1 for _ in range(4)]

    if print_params:
        print(
            'RLE params:', num, 'values', word_size, 'word_size', rle_sizes, 'rle_sizes'
        )

    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = input.read(1)
        j = i + 1 + input.read(rle_sizes[input.read(2)])
        if x:
            val = input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = input.read(word_size)
                out[i] = val
                i += 1
    return out
"""END FROM"""


def run(images_dir: pathlib.Path, raw_labels_dir: pathlib.Path, dataset_dir: pathlib.Path, classification_indices: Dict[str, int]):
    raw_label_names = glob.glob("*.json", root_dir=raw_labels_dir)
    raw_label_paths = [raw_labels_dir / name for name in raw_label_names]

    for raw_label_path in raw_label_paths:
        with open(raw_label_path, "r") as fp:
            data = json.load(fp)

        for annotation_data in data:
            # id = annotation_data["annotation_id"]

            if "bounding_box" not in annotation_data:
                continue

            image_name_raw = annotation_data["image"]
            img_name = re.search(r"\/([^\/]+\.png)", image_name_raw).group(1)
            id = img_name.split(".")[0]

            img_path = images_dir / img_name
            img_pil = Image.open(img_path)
            img = T.ToTensor()(img_pil)

            img_w = int(img.size(2))
            img_h = int(img.size(1))

            seg = torch.full((img_h, img_w), fill_value=255, dtype=torch.uint8)

            n_detections = len(annotation_data["bounding_box"])
            detection_is = {}

            valid = torch.full((n_detections,), fill_value=True, dtype=torch.bool)
            classifications = torch.zeros(n_detections, dtype=torch.long)
            bounding_boxes = torch.zeros((n_detections, 4), dtype=torch.float)

            for detection_i, bounding_box_data in enumerate(annotation_data["bounding_box"]):
                label = bounding_box_data["rectanglelabels"][0]
                x = bounding_box_data["x"]
                y = bounding_box_data["y"]
                w = bounding_box_data["width"]
                h = bounding_box_data["height"]

                classifications[detection_i] = classification_indices[label]
                bounding_boxes[detection_i] = torch.Tensor([(y + h / 2) / 100, (x + w / 2) / 100, h / 100, w / 100])

                detection_is[classification_indices[label]] = detection_i


            if "mask" in annotation_data:
                for mask_data in annotation_data["mask"]:
                    label = mask_data["brushlabels"][0]

                    mask_rle = mask_data["rle"]

                    detection_i = detection_is[classification_indices[label]]

                    mask_np = decode_rle(mask_rle).reshape((img_h, img_w, 4))[:, :, -1]
                    mask = torch.Tensor(mask_np / 255)

                    seg = torch.where(
                        mask == 1,
                        detection_i,
                        seg
                    )

            sample = SegmentationSample(
                img=img,
                seg=seg,
                valid=valid,
                classifications=classifications,
                bounding_boxes=bounding_boxes,
                img_valid=torch.ones_like(img, dtype=torch.bool)
            )

            sample.save(dataset_dir, id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")

    args = parser.parse_args()

    images_dir = pathlib.Path(args.dir).expanduser() / "images"
    raw_labels_dir = pathlib.Path(args.dir).expanduser() / "raw_labels"
    dataset_dir = pathlib.Path(args.dir).expanduser() / "all"

    assert images_dir.is_dir()
    assert raw_labels_dir.is_dir()

    # assert not dataset_dir.exists()

    # dataset_dir.mkdir()

    classification_indices = {
        "torpedo_22_bootlegger_circle": 0,
        "torpedo_22_bootlegger_trapezoid": 1,
    }

    run(images_dir, raw_labels_dir, dataset_dir, classification_indices)


if __name__ == "__main__":
    main()