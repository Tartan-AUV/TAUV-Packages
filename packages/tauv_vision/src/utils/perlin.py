import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from albumentations.core.transforms_interface import ImageOnlyTransform

# From https://pvigier.github.io/2018/06/13/perlin-noise-numpy.html
def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence

    return noise


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

def streaks(size, res, octaves):
    noise = generate_fractal_noise_2d((1024, 1024), res, octaves)

    noise = rotate(noise, random.uniform(0, 360))

    noise = crop(noise, size)

    return noise


class Streaks(ImageOnlyTransform):

    def __init__(self, shape, intensity, always_apply=False, p=0.5):
        super().__init__(always_apply, p)

        self.intensity = intensity

        self._streak_maps = [
            streaks(shape, (16, 16), octaves=2) for _ in range(32)
        ]

    def apply(self, old_img, **params):
        streak_map = random.sample(self._streak_maps, k=1)[0]

        intensity = random.randint(self.intensity[0], self.intensity[1])

        img = np.clip(old_img - intensity * streak_map[:, :, np.newaxis], 0, 255).astype(np.uint8)

        return img


# # Example usage:
# width, height = 640, 360
# for i in range(10):
#     noise = generate_fractal_noise_2d((1024, 1024), (2 ** random.randint(2, 4), 1), octaves=2)
#
#     noise = rotate(noise, random.uniform(0, 360))
#
#     noise = crop(noise, (height, width))
#
#     plt.imshow(noise)
#     plt.show()