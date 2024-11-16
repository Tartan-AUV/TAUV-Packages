import torch
import json
from PIL import Image
from dataclasses import dataclass
from typing import Optional, List, Dict, Callable
from pathlib import Path
from enum import Enum
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

from tauv_vision.yolo_pose.model.boxes import corners_to_box, box_to_corners

def closest_rotation_matrix(matrix):
    U, _, Vt = np.linalg.svd(matrix)
    R = np.dot(U, Vt)
    return R

class FallingThingsVariant(Enum):
    SINGLE = "single"
    MIXED = "mixed"


class FallingThingsEnvironment(Enum):
    Kitchen0 = "kitchen_0"
    Kitchen1 = "kitchen_1"
    Kitchen2 = "kitchen_2"
    Kitchen3 = "kitchen_3"
    Kitchen4 = "kitchen_4"
    KiteDemo0 = "kitedemo_0"
    KiteDemo1 = "kitedemo_1"
    KiteDemo2 = "kitedemo_2"
    KiteDemo3 = "kitedemo_3"
    KiteDemo4 = "kitedemo_4"
    Temple0 = "temple_0"
    Temple1 = "temple_1"
    Temple2 = "temple_2"
    Temple3 = "temple_3"
    Temple4 = "temple_4"


class FallingThingsObject(Enum):
    MasterChefCan = "002_master_chef_can_16k"
    CrackerBox = "003_cracker_box_16k"
    SugarBox = "004_sugar_box_16k"
    TomatoSoupCan = "005_tomato_soup_can_16k"
    MustardBottle = "006_mustard_bottle_16k"
    TunaFishCan = "007_tuna_fish_can_16k"
    PuddingBox = "008_pudding_box_16k"
    GelatinBox = "009_gelatin_box_16k"
    PottedMeatCan = "010_potted_meat_can_16k"
    Banana = "011_banana_16k"
    PitcherBase = "019_pitcher_base_16k"
    BleachCleanser = "021_bleach_cleanser_16k"
    Bowl = "024_bowl_16k"
    Mug = "025_mug_16k"
    PowerDrill = "035_power_drill_16k"
    WoodBlock = "036_wood_block_16k"
    Scissors = "037_scissors_16k"
    LargeMarker = "040_large_marker_16k"
    LargeClamp = "051_large_clamp_16k"
    ExtraLargeClamp = "052_extra_large_clamp_16k"
    FoamBrick = "061_foam_brick_16k"


falling_things_object_ids = {member.value: index + 1 for index, member in enumerate(FallingThingsObject)}


@dataclass
class FallingThingsSample:
    intrinsics: torch.Tensor
    valid: torch.Tensor
    classifications: torch.Tensor
    bounding_boxes: torch.Tensor
    camera_pose: torch.Tensor
    poses: torch.Tensor
    cuboids: torch.Tensor
    projected_cuboids: torch.Tensor
    img: torch.Tensor
    seg_map: torch.Tensor
    depth_map: torch.Tensor


class FallingThingsDataset(Dataset):
    """
    Directory structure, from https://pytorch.org/vision/0.15/_modules/torchvision/datasets/_stereo_matching.html#FallingThingsStereo

    <dir>
        single
            dir1
                scene1
                    _object_settings.json
                    _camera_settings.json
                    image1.left.depth.png
                    image1.right.depth.png
                    image1.left.jpg
                    image1.right.jpg
                    image2.left.depth.png
                    image2.right.depth.png
                    image2.left.jpg
                    image2.right
                    ...
                scene2
            ...
        mixed
            scene1
                _object_settings.json
                _camera_settings.json
                image1.left.depth.png
                image1.right.depth.png
                image1.left.jpg
                image1.right.jpg
                image2.left.depth.png
                image2.right.depth.png
                image2.left.jpg
                image2.right
                ...
            scene2
    """

    def __init__(self,
                 root: str,
                 variant: FallingThingsVariant,
                 environments: List[FallingThingsEnvironment],
                 objects: Optional[List[FallingThingsObject]],
                 transforms: Callable[[FallingThingsSample], FallingThingsSample],
                 ):
        super().__init__()

        self._root: Path = Path(root)
        self._variant: FallingThingsVariant = variant
        self._environments = environments

        if variant != FallingThingsVariant.SINGLE and objects is not None:
            raise ValueError("objects must be specified for variant SINGLE and cannot be specified for variant MIXED")

        self._objects = objects

        variant_dir = (self._root / self._variant.value).expanduser()

        if (not variant_dir.exists()) or (not variant_dir.is_dir()):
            raise ValueError(f"{variant_dir} does not exist")

        if variant == FallingThingsVariant.SINGLE:
            assert objects is not None
            object_dirs = [variant_dir / obj.value for obj in objects]
        elif variant == FallingThingsVariant.MIXED:
            object_dirs = [variant_dir]

        environment_dirs = []
        for object_dir in object_dirs:
            environment_dirs.extend([object_dir / environment.value for environment in environments])

        id_paths = self._get_id_paths(environment_dirs)
        id_paths = [value for sublist in id_paths.values() for value in sublist]
        self._id_paths: List[Path] = id_paths

    def __len__(self) -> int:
        return len(self._id_paths)

    def __getitem__(self, i: int) -> (torch.Tensor, ...):
        id_path = self._id_paths[i]

        print(f"loading {id_path}...")

        camera_json_path = id_path.with_name("_camera_settings.json")
        object_json_path = id_path.with_name("_object_settings.json")

        left_json_path = id_path.with_suffix(".left.json")
        left_img_path = id_path.with_suffix(".left.jpg")
        left_seg_path = id_path.with_suffix(".left.seg.png")
        left_depth_path = id_path.with_suffix(".left.depth.png")

        camera_data = self._get_json(camera_json_path)
        object_data = self._get_json(object_json_path)
        left_data = self._get_json(left_json_path)

        if len(left_data["objects"]) == 0:
            # Return the empty sample
            # TODO: FIX THIS
            return self[(i + 1) % len(self)]

        to_tensor = transforms.ToTensor()

        intrinsics = [
            camera_data["camera_settings"][0]["intrinsic_settings"]["fx"],
            camera_data["camera_settings"][0]["intrinsic_settings"]["fy"],
            camera_data["camera_settings"][0]["intrinsic_settings"]["cx"],
            camera_data["camera_settings"][0]["intrinsic_settings"]["cy"],
        ]
        intrinsics = torch.Tensor(intrinsics)

        classifications = [
            falling_things_object_ids[object["class"].lower()] for object in left_data["objects"]
        ]
        classifications = torch.Tensor(classifications).to(torch.long)

        valid = classifications > 0

        corners = [
            object["bounding_box"]["top_left"] + object["bounding_box"]["bottom_right"] for object in
            left_data["objects"]
        ]
        corners = torch.Tensor(corners)

        cuboids = [
            object["cuboid"] for object in left_data["objects"]
        ]
        cuboids = torch.Tensor(cuboids)

        camera_pose = left_data["camera_data"]["location_worldframe"] + left_data["camera_data"]["quaternion_xyzw_worldframe"]
        camera_pose = torch.Tensor(camera_pose)
        camera_pose[0:3] /= 100

        poses = [
            object["location"] + object["quaternion_xyzw"] for object in left_data["objects"]
        ]
        poses = torch.Tensor(poses)
        poses[:, 0:3] /= 100

        # Apply inverse of camera_pose to poses

        img = to_tensor(Image.open(left_img_path))
        seg = to_tensor(Image.open(left_seg_path))[0]
        seg = (255 * seg).to(torch.uint8)
        depth = to_tensor(Image.open(left_depth_path))[0]

        for object in object_data["exported_objects"]:
            seg = torch.where(seg == object["segmentation_class_id"],
                              falling_things_object_ids[object["class"].lower()], seg)

        depth = depth.float()
        depth = depth / 1e4

        corners[:, 0] = corners[:, 0] / img.size(1)
        corners[:, 1] = corners[:, 1] / img.size(2)
        corners[:, 2] = corners[:, 2] / img.size(1)
        corners[:, 3] = corners[:, 3] / img.size(2)
        bounding_boxes = corners_to_box(corners.unsqueeze(0)).squeeze(0)

        projected_cuboids = [
            object["projected_cuboid"] for object in left_data["objects"]
        ]
        projected_cuboids = torch.Tensor(projected_cuboids).flip(-1)
        centers = bounding_boxes[:, 0:2] * torch.Tensor([img.size(1), img.size(2)]).unsqueeze(0)
        projected_cuboids = torch.cat((centers.unsqueeze(1), projected_cuboids), dim=1)

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
            seg_map=seg,
            depth_map=depth,
            # position_map=position_map,
        )

        return sample

    def _get_id_paths(self, dirs: List[Path]) -> Dict[Path, List[int]]:
        id_paths = {}

        for dir in dirs:
            filenames = [file.name for file in dir.iterdir() if file.is_file()]

            unique_id_paths = set()
            for filename in filenames:
                if len(filename) >= 6 and filename[:6].isdigit():
                    unique_id_paths.add(dir / filename[:6])

            id_paths[dir] = list(unique_id_paths)

        return id_paths

    def _get_json(self, path: Path) -> Dict:
        with open(path, "r") as file:
            return json.load(file)


def main():
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    single_dataset = FallingThingsDataset(
        "~/Documents/falling_things/fat",
        FallingThingsVariant.SINGLE,
        [
            FallingThingsEnvironment.Kitchen0,
            FallingThingsEnvironment.Kitchen1,
            FallingThingsEnvironment.Kitchen2,
            FallingThingsEnvironment.Kitchen3,
            FallingThingsEnvironment.Kitchen4,
        ],
        [
            FallingThingsObject.CrackerBox,
        ],
        transforms=lambda x: x,
    )

    mixed_dataset = FallingThingsDataset(
        "~/Documents/falling_things/fat",
        FallingThingsVariant.MIXED,
        [
            FallingThingsEnvironment.Kitchen0,
            FallingThingsEnvironment.Kitchen1,
            FallingThingsEnvironment.Kitchen2,
            FallingThingsEnvironment.Kitchen3,
            FallingThingsEnvironment.Kitchen4,
        ],
        None,
        transforms=lambda x: x,
    )

    mixed_sample = mixed_dataset[0]
    fig, axs = plt.subplots()
    axs.imshow(mixed_sample.img.permute(1, 2, 0))
    axs.scatter(mixed_sample.projected_cuboids[:, :, 0], mixed_sample.projected_cuboids[:, :, 1])

    bbox = box_to_corners(mixed_sample.bounding_boxes.unsqueeze(0)).squeeze(0)
    for i in range(bbox.size(0)):
        rectangle = patches.Rectangle(
            (bbox[i, 1], bbox[i, 0]),
            bbox[i, 3] - bbox[i, 1],
            bbox[i, 2] - bbox[i, 0],
            linewidth=2,
            edgecolor="b",
            facecolor="none",
        )
        axs.add_patch(rectangle)
    plt.figure()
    plt.imshow(mixed_sample.seg_map)
    plt.figure()
    plt.imshow(mixed_sample.depth_map)

    x, y, z = mixed_sample.position_map

    plt.figure()
    plt.imshow(torch.where(x == 0, torch.nan, x))
    plt.figure()
    plt.imshow(torch.where(y == 0, torch.nan, y))
    plt.figure()
    plt.imshow(torch.where(z == 0, torch.nan, z))

    plt.show()


def get_position_map(camera_pose: torch.Tensor,
                     poses: torch.Tensor,
                     classifications: torch.Tensor,
                     seg_map: torch.Tensor,
                     depth_map: torch.Tensor,
                     intrinsics: torch.Tensor,
                     ) -> torch.Tensor:
    import matplotlib.pyplot as plt
    # Shape n_detections x 3 x h x w

    n_detections = poses.size()[0]
    h, w = depth_map.size()

    position_map = torch.zeros((3, h, w), dtype=torch.float, device=poses.device)

    for detection_i in range(n_detections):
        pose = poses[detection_i]

        cam_z = depth_map.reshape(-1)
        cam_pixel_x = torch.arange(0, w).repeat(h)
        cam_pixel_y = torch.arange(0, h).repeat_interleave(w)
        f_x, f_y, c_x, c_y = intrinsics
        cam_x = (cam_z / f_x) * (cam_pixel_x - c_x)
        cam_y = (cam_z / f_y) * (cam_pixel_y - c_y)

        cam_pos = torch.stack((cam_x, cam_y, cam_z), dim=0)

        cam_t_obj_trans = pose[0:3]
        cam_t_obj_quat_xyzw = pose[3:7]

        cam_t_obj_rotm = quat_xyzw_to_rotm(cam_t_obj_quat_xyzw)

        obj_t_cam_rotm = torch.transpose(cam_t_obj_rotm, dim0=0, dim1=1)
        obj_t_cam_trans = -torch.matmul(obj_t_cam_rotm, cam_t_obj_trans)

        obj_pos = torch.matmul(obj_t_cam_rotm, cam_pos) + obj_t_cam_trans.unsqueeze(1)

        obj_pos = obj_pos.reshape(3, h, w)

        position_map = torch.where(seg_map == classifications[detection_i], obj_pos, position_map)

    return position_map

def quat_xyzw_to_rotm(quat_xyzw: torch.Tensor) -> torch.Tensor:
    q_x, q_y, q_z, q_w = quat_xyzw
    rotm = torch.tensor([
        [1 - 2 * q_y ** 2 - 2 * q_z ** 2, 2 * q_x * q_y - 2 * q_z * q_w, 2 * q_x * q_z + 2 * q_y * q_w],
        [2 * q_x * q_y + 2 * q_z * q_w, 1 - 2 * q_x ** 2 - 2 * q_z ** 2, 2 * q_y * q_z - 2 * q_x * q_w],
        [2 * q_x * q_z - 2 * q_y * q_w, 2 * q_y * q_z + 2 * q_x * q_w, 1 - 2 * q_x ** 2 - 2 * q_y ** 2]
    ])
    return rotm


if __name__ == "__main__":
    main()

