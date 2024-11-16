import torch
from pathlib import Path
from tauv_vision.yolact.model.config import ModelConfig, TrainConfig, ClassConfig, ClassConfigSet
from tauv_vision.datasets.segmentation_dataset.segmentation_dataset import SegmentationDataset, SegmentationSample, \
    SegmentationDatasetSet
from tauv_vision.utils.plot import save_plot, plot_prototype, plot_mask, plot_detection
import matplotlib.pyplot as plt


class_config = ClassConfigSet([
    ClassConfig(
        id="torpedo_22_gman",
        index=1,
    ),
    ClassConfig(
        id="torpedo_22_bootlegger",
        index=2,
    ),
    ClassConfig(
        id="torpedo_22_circle",
        index=3,
    ),
    ClassConfig(
        id="torpedo_22_star",
        index=4,
    ),
    ClassConfig(
        id="torpedo_22_trapezoid",
        index=5,
    ),
    ClassConfig(
        id="buoy_23",
        index=6,
    ),
    ClassConfig(
        id="buoy_23_abydos_1",
        index=7,
    ),
    ClassConfig(
        id="buoy_23_abydos_2",
        index=8,
    ),
    ClassConfig(
        id="buoy_23_earth_1",
        index=9,
    ),
    ClassConfig(
        id="buoy_23_earth_2",
        index=10,
    ),
])


def main():
    dataset_root = Path("~/Documents/TAUV-Datasets/believe-special-kind").expanduser()
    class_ids_to_indices = {c.id: c.index for c in class_config.configs}

    dataset = SegmentationDataset(dataset_root, SegmentationDatasetSet.TRAIN, class_ids_to_indices)

    for sample in dataset:
        detection_fig = plot_detection(
            sample.img,
            torch.zeros((0,)),
            torch.zeros((0, 4)),
            sample.valid,
            sample.classifications,
            sample.bounding_boxes,
        )

        plt.show()
        plt.close(detection_fig)

        pass


if __name__ == "__main__":
    main()