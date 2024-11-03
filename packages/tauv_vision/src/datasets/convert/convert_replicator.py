import argparse
from pathlib import Path
from typing import List, Set
import numpy as np
from math import pi
import glob
import re
from PIL import Image
from tqdm import tqdm
import json
import random
import dirhash
import datetime
from spatialmath import SE3, SO3
import human_id


def wrap(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi


def orthonormalize(R: np.array) -> np.array:
    R = R.astype(np.float64)

    x = R[:, 0]
    y = R[:, 1]
    z = R[:, 2]

    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    z = z / np.linalg.norm(z)

    error_xy = 0.5 * np.dot(x, y)
    error_yz = 0.5 * np.dot(y, z)
    error_zx = 0.5 * np.dot(z, x)

    R_norm = np.vstack((
        x - error_xy * y - error_zx * z,
        y - error_xy * x - error_yz * z,
        z - error_zx * x - error_yz * y,
    )).T

    return R_norm

"""
Example seg output:

...
bounding_box_2d_tight_0_9999.npy
bounding_box_2d_tight_labels_0_9999.json
bounding_box_2d_tight_prim_paths_0_9999.json
instance_segmentation_0_9999.png
instance_segmentation_mapping_0_9999.json
instance_segmentation_semantics_mapping_0_9999.json
rgb_0_9999.png
...

sample id will be content between "rgb_" and ".png". So "0_9999"
"""

# camera_base is oriented the same way as the Omniverse default frame
#   x right, y up, z in
# camera is oriented like a standard camera frame
#   x right, y down, z out

cam_base_t_cam = SE3(SO3.TwoVectors(x="x", y="-y"))

# rvec, _ = cv2.Rodrigues(cam_t_object.R)
# tvec = cam_t_object.t
# axis_img = np.flip(cv2.drawFrameAxes(np.flip(img, axis=-1).copy(), camera_projection_np[:, 0:3], np.zeros((4,)), rvec, tvec, 0.1, 5), axis=-1)


def get_sample_ids(replicator_out_dir: Path) -> List[str]:
    rgb_names = glob.glob("rgb*", root_dir=replicator_out_dir)

    sample_id_re = re.compile(r"(?<=rgb_)(.*?)(?=\.png)")

    sample_ids = []
    for rgb_name in rgb_names:
        match = re.search(sample_id_re, rgb_name)
        if match is not None:
            sample_ids.append(match.group(1))
        else:
            raise ValueError("malformed rgb file name: {rgb_name}")

    return sample_ids


def split(pop: List, splits: List[float]) -> List[List]:
    out_splits = []

    pop_size = len(pop)
    for split in splits[:-1]:
        out_split = random.sample(pop, int(split * pop_size))
        pop = [x for x in pop if x not in out_split]
        out_splits.append(out_split)

    out_splits.append(pop)

    return out_splits


def convert_sample(replicator_out_dir: Path, dataset_dir: Path, sample_id: str) -> Set[str]:
    rgb_path = replicator_out_dir / f"rgb_{sample_id}.png"

    bbox_path = replicator_out_dir / f"bounding_box_2d_tight_{sample_id}.npy"
    bbox_class_path = replicator_out_dir / f"bounding_box_2d_tight_labels_{sample_id}.json"
    bbox_instance_path = replicator_out_dir / f"bounding_box_2d_tight_prim_paths_{sample_id}.json"

    bbox_3d_path = replicator_out_dir / f"bounding_box_3d_{sample_id}.npy"
    bbox_3d_instance_path = replicator_out_dir / f"bounding_box_3d_prim_paths_{sample_id}.json"

    seg_path = replicator_out_dir / f"instance_segmentation_{sample_id}.png"
    seg_instance_path = replicator_out_dir / f"instance_segmentation_mapping_{sample_id}.json"

    camera_path = replicator_out_dir / f"camera_params_{sample_id}.json"
    if not camera_path.exists():
        camera_path = replicator_out_dir / f"camera_params_0_0000.json"

    rgb_out_path = dataset_dir / "data" / f"{sample_id}.png"
    seg_out_path = dataset_dir / "data" / f"{sample_id}_seg.png"
    json_out_path = dataset_dir / "data" / f"{sample_id}.json"

    rgb_raw_pil = Image.open(rgb_path)
    seg_raw_pil = Image.open(seg_path)

    bboxes_raw = np.load(bbox_path)
    with open(bbox_class_path, "r") as fp:
        bbox_classes_raw = json.load(fp)
    with open(bbox_instance_path, "r") as fp:
        bbox_instances_raw = json.load(fp)
    with open(seg_instance_path, "r") as fp:
        seg_instances_raw = json.load(fp)

    bboxes_3d_raw = np.load(bbox_3d_path)
    with open(bbox_3d_instance_path, "r") as fp:
        bbox_3d_instances_raw = json.load(fp)

    with open(camera_path, "r") as fp:
        camera_raw = json.load(fp)

    seg_instances_raw = {v: k for k, v in seg_instances_raw.items()}

    n_objects = len(bboxes_raw)

    w, h = rgb_raw_pil.size

    world_units_to_m = camera_raw["metersPerSceneUnit"]

    # https://www.songho.ca/opengl/gl_projectionmatrix.html
    # https://forums.developer.nvidia.com/t/get-object-center-pose-in-camera-frame-viewport/255358/8
    camera_projection_raw_np = np.array(camera_raw["cameraProjection"]).reshape((4, 4)).T
    camera_projection_np = np.array([
        [camera_projection_raw_np[0, 0] * w / 2, 0, w / 2, 0],
        [0, camera_projection_raw_np[1, 1] * h / 2, h / 2, 0],
        [0, 0, 1, 0],
    ])

    world_t_camera_base_np = np.array(camera_raw["cameraViewTransform"]).reshape(4, 4).T.astype(np.float64)
    world_t_camera_base_np[0:3, 0:3] = orthonormalize(world_t_camera_base_np[0:3, 0:3])
    world_t_camera_base_np[0:3, 3] *= world_units_to_m
    world_t_cam_base = SE3(world_t_camera_base_np)
    # world_t_cam_base = SE3(SO3.TwoVectors(x="y", y="z")) * SE3(world_t_camera_base_np)

    # TODO: THIS MUST BE FIXed

    # world_t_weird_world = SE3(SO3.TwoVectors(x="-z", y="x"))
    # world_t_cam_base = world_t_weird_world * SE3(world_t_camera_base_np)

    # world_t_cam_base = SE3.Rt(
    #     SO3.TwoVectors(y="-z", x="x"),
    #     np.array([0, 8, 0]),
    # )

    # TODO: world_t_cam_base is not what i expect

    objects = []

    seg_raw_np = np.array(seg_raw_pil)
    seg_np = np.full((h, w), fill_value=255, dtype=np.uint8)

    class_ids = set()

    for object_i in range(n_objects):
        bbox_class_index, x0, y0, x1, y1, occlusion = bboxes_raw[object_i]

        if bbox_instances_raw[object_i] not in bbox_3d_instances_raw:
            continue

        bbox_3d_i = bbox_3d_instances_raw.index(bbox_instances_raw[object_i])

        bbox_class_index_3d, x0_3d, y0_3d, z0_3d, x1_3d, y1_3d, z1_3d, transform_3d, occlusion_3d = bboxes_3d_raw[bbox_3d_i]

        assert bbox_class_index == bbox_class_index_3d

        bbox_x = ((x0 + x1) / 2) / w
        bbox_y = ((y0 + y1) / 2) / h

        bbox_w = abs(x1 - x0) / w
        bbox_h = abs(y1 - y0) / h

        bbox_class_id = bbox_classes_raw[str(bbox_class_index)]["class"].split(",")[-1]

        if bbox_instances_raw[object_i] in seg_instances_raw:
            seg_value = int(seg_instances_raw[bbox_instances_raw[object_i]])

            seg_np = np.where(
                seg_raw_np == seg_value,
                object_i,
                seg_np,
            )

        world_t_object_np = transform_3d.transpose().astype(np.float64)
        world_t_object_np[:, 0:3] = world_t_object_np[:, 0:3] / np.linalg.norm(world_t_object_np[:, 0:3], axis=0)
        world_t_object_np[0:3, 3] *= world_units_to_m
        world_t_object_np[0:3, 0:3] = orthonormalize(world_t_object_np[0:3, 0:3])
        world_t_object = SE3(world_t_object_np)

        cam_t_object = cam_base_t_cam.inv() * world_t_cam_base.inv() * world_t_object

        p0_3d_object = world_units_to_m * np.array([x0_3d, y0_3d, z0_3d])
        p1_3d_object = world_units_to_m * np.array([x1_3d, y1_3d, z1_3d])

        p0_3d_cam = (cam_t_object * p0_3d_object).flatten()
        p1_3d_cam = (cam_t_object * p1_3d_object).flatten()

        objects.append({
            "label": bbox_class_id,
            "visibility": round(1 - occlusion, 4),
            "bbox": {
                "y": round(bbox_y, 4),
                "x": round(bbox_x, 4),
                "h": round(bbox_h, 4),
                "w": round(bbox_w, 4),
            },
            "bbox_3d": {
                "x0": round(p0_3d_cam[0], 4), # These are in object frame. Might not be what we want.
                "y0": round(p0_3d_cam[1], 4),
                "z0": round(p0_3d_cam[2], 4),
                "x1": round(p1_3d_cam[0], 4),
                "y1": round(p1_3d_cam[1], 4),
                "z1": round(p1_3d_cam[2], 4),
            },
            "pose": {
                "x": round(cam_t_object.t[0], 4),
                "y": round(cam_t_object.t[1], 4),
                "z": round(cam_t_object.t[2], 4),
                "distance": round(np.linalg.norm(cam_t_object.t), 4),
                "roll": round(wrap(cam_t_object.rpy()[2]), 4),
                "pitch": round(wrap(cam_t_object.rpy()[1]), 4),
                "yaw": round(wrap(cam_t_object.rpy()[0]), 4),
                "cam_t_object": [round(x, 12) for x in np.array(cam_t_object).flatten()],
            }
        })

        class_ids.add(bbox_class_id)

    seg_pil = Image.fromarray(seg_np)

    camera = {
        "fy": round(camera_projection_np[1, 1], 4),
        "fx": round(camera_projection_np[0, 0], 4),
        "cy": round(camera_projection_np[1, 2], 4),
        "cx": round(camera_projection_np[0, 2], 4),
        "h": int(camera_raw["renderProductResolution"][1]),
        "w": int(camera_raw["renderProductResolution"][0]),
        "projection": [round(x, 12) for x in camera_projection_np.flatten()],
    }

    json_data = {
        "camera": camera,
        "objects": objects,
    }

    rgb_raw_pil.save(rgb_out_path)
    seg_pil.save(seg_out_path)

    with open(json_out_path, "w") as fp:
        json.dump(json_data, fp, indent="  ")

    return class_ids


def convert(replicator_out_dir: Path, datasets_dir: Path, splits: List[float], email: str, description: str):
    if not np.isclose(sum(splits), 1):
        raise ValueError(f"Error: splits must sum to 1")

    if not replicator_out_dir.exists() or not replicator_out_dir.is_dir():
        raise ValueError(f"Error: {replicator_out_dir} does not exist")

    if not datasets_dir.exists() or not datasets_dir.is_dir():
        raise ValueError(f"Error: {datasets_dir} does not exist")

    dataset_id = human_id.generate_id(word_count=3)

    dataset_dir = datasets_dir / dataset_id

    meta_json_path = dataset_dir / "meta.json"
    splits_json_path = dataset_dir / "splits.json"
    classes_json_path = dataset_dir / "classes.json"

    if dataset_dir.exists():
        raise ValueError(f"Error: {dataset_dir} already exists")

    print(f"Creating dataset {dataset_dir}...")
    print(f"Input: {replicator_out_dir}")
    print(f"Author: {email}")
    print(f"Description: {description}")

    dataset_dir.mkdir()
    (dataset_dir / "data").mkdir()

    sample_ids = get_sample_ids(replicator_out_dir)

    class_ids = set()

    for sample_id in tqdm(sample_ids):
        new_class_ids = convert_sample(replicator_out_dir, dataset_dir, sample_id)
        class_ids = class_ids.union(new_class_ids)

    sample_id_splits = split(sample_ids, splits)
    splits_json_data = {
        "splits": {
            "train": sample_id_splits[0],
            "val": sample_id_splits[1],
            "test": sample_id_splits[2]
        }
    }

    with open(splits_json_path, "w") as fp:
        json.dump(splits_json_data, fp, indent="  ")

    classes_json_data = {
        "classes": [{"id": class_id} for class_id in class_ids]
    }

    with open(classes_json_path, "w") as fp:
        json.dump(classes_json_data, fp, indent="  ")

    md5 = dirhash.dirhash(dataset_dir, "md5")

    meta_json_data = {
        "author": email,
        "has_seg": True,
        "has_pose": True,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "description": description,
        "md5": md5,
    }

    with open(meta_json_path, "w") as fp:
        json.dump(meta_json_data, fp, indent="  ")

    print(f"Created dataset {dataset_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replicator_out_dir")
    parser.add_argument("datasets_dir")
    parser.add_argument("--splits", type=float, nargs=3, required=True)
    parser.add_argument("--email", type=str, required=True)
    parser.add_argument("--description", type=str, required=True)

    args = parser.parse_args()

    replicator_out_dir = Path(args.replicator_out_dir).expanduser()
    datasets_dir = Path(args.datasets_dir).expanduser()

    convert(replicator_out_dir, datasets_dir, args.splits, args.email, args.description)


if __name__ == "__main__":
    main()