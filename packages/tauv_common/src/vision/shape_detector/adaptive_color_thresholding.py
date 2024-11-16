import numpy as np
from dataclasses import dataclass
import sys
import cv2

@dataclass
class GetAdaptiveColorThresholdingParams:
    global_thresholds: np.array
    local_thresholds: np.array
    window_size: int


def get_adaptive_color_thresholding(img: np.array, params: GetAdaptiveColorThresholdingParams) -> np.array:
    global_thresholds = params.global_thresholds
    local_thresholds = params.local_thresholds
    window_size = params.window_size

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = np.ones((hsv_img.shape[0], hsv_img.shape[1]), dtype=np.uint8)
    global_mask = np.zeros((hsv_img.shape[0], hsv_img.shape[1]), dtype=np.uint8)
    local_mask = np.zeros((hsv_img.shape[0], hsv_img.shape[1]), dtype=np.uint8)

    for global_threshold in global_thresholds:
        global_h_mask = (hsv_img[:, :, 0] >= global_threshold[0]) & (hsv_img[:, :, 0] <= global_threshold[3])
        global_s_mask = (hsv_img[:, :, 1] >= global_threshold[1]) & (hsv_img[:, :, 1] <= global_threshold[4])
        global_v_mask = (hsv_img[:, :, 2] >= global_threshold[2]) & (hsv_img[:, :, 2] <= global_threshold[5])

        global_mask = global_mask | (global_h_mask & global_s_mask & global_v_mask)

    mask = mask & global_mask

    # blurred_hsv_img = cv2.GaussianBlur(hsv_img, (window_size, window_size), 0)
    blurred_hsv_img = cv2.blur(hsv_img, (window_size, window_size))

    for local_threshold in local_thresholds:
        local_s_mask = (hsv_img[:, :, 1] >= (blurred_hsv_img[:, :, 1].astype(np.float32) + local_threshold[0])) & \
                       (hsv_img[:, :, 1] <= (blurred_hsv_img[:, :, 1].astype(np.float32) + local_threshold[2]))
        local_v_mask = (hsv_img[:, :, 2] >= (blurred_hsv_img[:, :, 2].astype(np.float32) + local_threshold[1])) & \
                       (hsv_img[:, :, 2] <= (blurred_hsv_img[:, :, 2].astype(np.float32) + local_threshold[3]))

        local_mask = local_mask | (local_s_mask & local_v_mask)

    mask = mask & local_mask

    return mask


def main():
    import sys
    import matplotlib.pyplot as plt

    img_path = sys.argv[1]

    img = cv2.imread(img_path)

    global_thresholds = np.array([
        [0, 0, 50, 20, 255, 150],
        [160, 0, 0, 180, 255, 255],
    ])

    local_thresholds = np.array([
        [20, -255, 255, 255],
    ])

    window_size = 35

    plt.figure()
    plt.imshow(img[:, :, ::-1])

    mask = get_adaptive_color_thresholding(img, global_thresholds, local_thresholds, window_size)

    plt.figure()
    plt.imshow(mask)

    plt.show()

if __name__ == "__main__":
    main()