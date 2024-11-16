import json
import yaml
import numpy as np
import cv2
import argparse

# Define mappings between Kalibr cameras and JSON IDs
camera_id_map = {
    "cam0": 0,
    "cam1": 1,
    "cam2": 2,
}

def load_yaml(filename):
    with open(filename, 'r') as file:
        return yaml.safe_load(file)

def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def save_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def convert_intrinsics(intrinsics):
    return [
        [intrinsics[0], 0, intrinsics[2]],
        [0, intrinsics[1], intrinsics[3]],
        [0, 0, 1]
    ]

def convert_distortion(distortion_coeffs):
    return distortion_coeffs[:5] + [0] * (14 - len(distortion_coeffs))

def to_transformation_matrix(rotation_matrix, translation_vector):
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector
    return transformation_matrix

def extract_extrinsics(transform):
    rotation_matrix = np.array([row[:3] for row in transform[:3]])
    translation_vector = np.array([row[3] for row in transform[:3]])
    return rotation_matrix, translation_vector

def chain_extrinsics(yaml_data):
    # Dictionary to store the chained extrinsics with respect to cam0
    chained_extrinsics = {}

    # Start with the identity transformation for cam0
    chained_extrinsics[0] = np.eye(4)

    # Chain transformations from cam0 to each subsequent camera
    for cam_name, cam_id in sorted(camera_id_map.items()):
        if cam_name == "cam0":
            continue

        cam_data = yaml_data[cam_name]
        prev_cam_name = f"cam{cam_id - 1}"
        prev_transform = chained_extrinsics[cam_id - 1]

        # Get relative transform from YAML data
        rotation_matrix, translation_vector = extract_extrinsics(cam_data["T_cn_cnm1"])
        rel_transform = to_transformation_matrix(rotation_matrix, translation_vector)

        # Chain relative transform to the previous one to get cam0-based transform
        chained_extrinsics[cam_id] = np.dot(prev_transform, rel_transform)

    return chained_extrinsics

def update_camera_data(json_data, yaml_data, chained_extrinsics):
    for cam_name, cam_data in yaml_data.items():
        camera_id = camera_id_map.get(cam_name)
        if camera_id is not None:
            for json_cam_entry_idx in range(len(json_data["cameraData"])):
                if json_data["cameraData"][json_cam_entry_idx][0] == camera_id:
                    json_cam_data = json_data["cameraData"][json_cam_entry_idx][1]

            # Update intrinsics
            json_cam_data["intrinsicMatrix"] = convert_intrinsics(cam_data["intrinsics"])
            json_cam_data["distortionCoeff"] = convert_distortion(cam_data["distortion_coeffs"])

            # Update resolution
            # json_cam_data["width"], json_cam_data["height"] = cam_data["resolution"]
            #
            # # Update extrinsics based on cam0 chaining
            # if camera_id in chained_extrinsics:
            #     transform_matrix = chained_extrinsics[camera_id]
            #     rotation_matrix = transform_matrix[:3, :3]
            #     translation_vector = transform_matrix[:3, 3]
            #
            #     json_cam_data["extrinsics"]["rotationMatrix"] = rotation_matrix.tolist()
            #     json_cam_data["extrinsics"]["translation"] = {
            #         "x": translation_vector[0],
            #         "y": translation_vector[1],
            #         "z": translation_vector[2]
            #     }

def compute_rectification(json_data):
    left_cam = json_data["cameraData"][1][1]
    right_cam = json_data["cameraData"][2][1]

    # Extract intrinsics and distortion coefficients
    left_intrinsics = np.array(left_cam["intrinsicMatrix"])
    left_distortion = np.array(left_cam["distortionCoeff"][:5])
    right_intrinsics = np.array(right_cam["intrinsicMatrix"])
    right_distortion = np.array(right_cam["distortionCoeff"][:5])

    # Extract extrinsic translation and rotation
    R = np.array(right_cam["extrinsics"]["rotationMatrix"])
    T = np.array([right_cam["extrinsics"]["translation"][k] for k in ("x", "y", "z")])

    # Compute rectification
    _, left_R, right_R, _, _, _, _ = cv2.stereoRectify(
        left_intrinsics, left_distortion, right_intrinsics, right_distortion,
        (left_cam["width"], left_cam["height"]), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    # Update rectification matrices in the JSON data
    json_data["stereoRectificationData"]["rectifiedRotationLeft"] = left_R.tolist()
    json_data["stereoRectificationData"]["rectifiedRotationRight"] = right_R.tolist()

def main(yaml_filename, json_filename, output_filename):
    yaml_data = load_yaml(yaml_filename)
    json_data = load_json(json_filename)

    # Chain extrinsics based on cam0 reference
    chained_extrinsics = chain_extrinsics(yaml_data)

    # Update camera data based on YAML input
    update_camera_data(json_data, yaml_data, chained_extrinsics)

    # Compute rectification matrices for stereo cameras
    compute_rectification(json_data)

    # Save updated JSON to new file
    save_json(json_data, output_filename)
    print(f"Updated JSON saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_filename")
    parser.add_argument("json_filename")
    parser.add_argument("output_filename")
    args = parser.parse_args()
    # Example usage:
    main(args.yaml_filename, args.json_filename, args.output_filename)


