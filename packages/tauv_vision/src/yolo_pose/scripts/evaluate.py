import torch
from pathlib import Path
from typing import List
import torch.nn.functional as F
from torch.utils.data import DataLoader

from yolo_pose.model.config import Config
from yolo_pose.model.model import YoloPose
from datasets.falling_things_dataset.falling_things_dataset import FallingThingsDataset, FallingThingsVariant, FallingThingsEnvironment, FallingThingsSample, FallingThingsObject


config = Config(
    in_w=960,
    in_h=480,
    feature_depth=256,
    n_classes=23,
    n_prototype_masks=32,
    n_masknet_layers_pre_upsample=1,
    n_masknet_layers_post_upsample=1,
    pointnet_layers=[
        (3, 6, 512),
        (7, 10, 128),
    ],
    pointnet_feature_depth=128,
    prototype_belief_depth=64,
    prototype_affinity_depth=64,
    belief_depth=9,
    affinity_depth=16,
    n_prediction_head_layers=1,
    n_fpn_downsample_layers=2,
    belief_sigma=4,
    affinity_radius=4,
    anchor_scales=(24, 48, 96, 192, 384),
    anchor_aspect_ratios=(1 / 2, 1, 2),
    iou_pos_threshold=0.5,
    iou_neg_threshold=0.4,
    negative_example_ratio=3,
)

falling_things_root = "~/Documents/falling_things/fat"
weights_root = "~/Documents/yolo_pose/weights"

test_environments = [
    FallingThingsEnvironment.Kitchen0,
    FallingThingsEnvironment.Kitchen1,
    FallingThingsEnvironment.Kitchen2,
    FallingThingsEnvironment.Kitchen3,
    FallingThingsEnvironment.Kitchen4,
]
test_objects = [
    FallingThingsObject.CrackerBox,
]


def collate_samples(samples: List[FallingThingsSample]) -> FallingThingsSample:
    n_detections = [sample.valid.size(0) for sample in samples]
    max_n_detections = max(n_detections)

    intrinsics = torch.stack([sample.intrinsics for sample in samples], dim=0)
    valid = torch.stack([
        F.pad(sample.valid, (0, max_n_detections - sample.valid.size(0)), value=False)
        for sample in samples
    ], dim=0)
    classifications = torch.stack([
        F.pad(sample.classifications, (0, max_n_detections - sample.classifications.size(0)), value=False)
        for sample in samples
    ], dim=0)
    bounding_boxes = torch.stack([
        F.pad(sample.bounding_boxes, (0, 0, 0, max_n_detections - sample.bounding_boxes.size(0)), value=False)
        for sample in samples
    ], dim=0)
    poses = torch.stack([
        F.pad(sample.poses, (0, 0, 0, max_n_detections - sample.poses.size(0)), value=False)
        for sample in samples
    ], dim=0)
    cuboids = torch.stack([
        F.pad(sample.cuboids, (0, 0, 0, 0, 0, max_n_detections - sample.cuboids.size(0)), value=False)
        for sample in samples
    ], dim=0)
    projected_cuboids = torch.stack([
        F.pad(sample.projected_cuboids, (0, 0, 0, 0, 0, max_n_detections - sample.projected_cuboids.size(0)), value=False)
        for sample in samples
    ], dim=0)
    camera_pose = torch.stack([sample.camera_pose for sample in samples], dim=0)
    img = torch.stack([sample.img for sample in samples], dim=0)
    seg_map = torch.stack([sample.seg_map for sample in samples], dim=0)
    depth_map = torch.stack([sample.depth_map for sample in samples], dim=0)

    sample = FallingThingsSample(
        intrinsics=intrinsics,
        valid=valid,
        classifications=classifications,
        bounding_boxes=bounding_boxes,
        camera_pose=camera_pose,
        poses=poses,
        cuboids=cuboids,
        projected_cuboids=projected_cuboids,
        img=img,
        seg_map=seg_map,
        depth_map=depth_map,
    )

    return sample


def main():
    model = YoloPose(config)
    model.load_state_dict(torch.load(Path(weights_root).expanduser() / "30.pt", map_location=torch.device("cpu")))

    test_dataset = FallingThingsDataset(
        falling_things_root,
        FallingThingsVariant.SINGLE,
        test_environments,
        test_objects,
        lambda x: x,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=collate_samples,
        shuffle=True,
    )

    model.eval()

    for batch in test_dataloader:
        prediction = model(batch.img)
        classification, box_encoding, mask_coeff, belief_coeff, affinity_coeff, anchor, mask_prototype, belief_prototype, affinity_prototype = prediction

        pass

if __name__ == "__main__":
    main()
