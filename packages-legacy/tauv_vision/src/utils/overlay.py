import numpy as np
import random
from PIL import Image
import cv2
from albumentations.core.transforms_interface import ImageOnlyTransform


def rotate(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result


def crop(image, size):
    y = random.randint(0, image.shape[0] - size[0] - 1)
    x = random.randint(0, image.shape[1] - size[1] - 1)

    result = image[y:y+size[0], x:x+size[1]]

    return result


class Overlay(ImageOnlyTransform):

    def __init__(self, paths, intensity, always_apply=False, p=0.5):
        super().__init__(always_apply, p)

        self.paths = paths
        self.intensity = intensity

        self.imgs = [np.array(Image.open(path).convert("RGB")) for path in self.paths]

    def apply(self, old_img_np, **params):
        intensity = random.uniform(self.intensity[0], self.intensity[1])
        angle = random.uniform(0, 360)
        overlay_img_np = random.choice(self.imgs)

        overlay_img_np = rotate(overlay_img_np, angle)
        overlay_img_np = cv2.resize(overlay_img_np, None, fx=random.uniform(0.5, 2), fy=random.uniform(0.5, 2))
        overlay_img_np = crop(overlay_img_np, (old_img_np.shape[0], old_img_np.shape[1]))

        img_np = np.clip(old_img_np.astype(np.int64) + intensity * (overlay_img_np.astype(np.int64) - 127), 0, 255).astype(np.uint8)

        return img_np

