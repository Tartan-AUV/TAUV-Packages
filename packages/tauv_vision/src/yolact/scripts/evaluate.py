import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image, ImageOps
import pathlib
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from tauv_vision.yolact.model.config import ModelConfig
from tauv_vision.yolact.model.model import Yolact
from tauv_vision.yolact.utils.plot import save_plot, plot_prototype, plot_mask, plot_detection
from tauv_vision.yolact.model.boxes import box_decode
from tauv_vision.yolact.model.masks import assemble_mask


config = ModelConfig(
    in_w=640,
    in_h=360,
    feature_depth=32,
    n_classes=3,
    n_prototype_masks=32,
    n_masknet_layers_pre_upsample=1,
    n_masknet_layers_post_upsample=1,
    n_classification_layers=0,
    n_box_layers=0,
    n_mask_layers=0,
    n_fpn_downsample_layers=2,
    anchor_scales=(24, 48, 96, 192, 384),
    anchor_aspect_ratios=(1 / 2, 1, 2),
    iou_pos_threshold=0.4,
    iou_neg_threshold=0.3,
    negative_example_ratio=3,
)

img_mean = (0.485, 0.456, 0.406)
img_stddev = (0.229, 0.224, 0.225)

img_path = pathlib.Path("/Volumes/Storage/bags/2023-09-05/all/1699208876839857139.png").expanduser()
weights_path = pathlib.Path("~/Documents/TAUV-Vision/weights/wandering-mountain-193_30.pt").expanduser()

def main():
    model = Yolact(config)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))

    model.eval()

    img_pil = Image.open(img_path).convert("RGB")
    img_np = np.array(img_pil)
    # img_np = np.flip(img_np, -1)

    val_transform = A.Compose(
        [
            A.Resize(height=360, width=640),
            A.Normalize(mean=img_mean, std=img_stddev),
            ToTensorV2(),
        ],
    )

    # img_raw = T.ToTensor()(img_pil).unsqueeze(0)
    # img_raw = img_raw.flip(1)
    # img = T.Normalize(mean=img_mean, std=img_stddev)(img_raw)

    img = val_transform(image=img_np)["image"].unsqueeze(0)

    plt.imshow(img[0].permute(1, 2, 0).detach().cpu())
    plt.show()

    prediction = model(img)
    classification, box_encoding, mask_coeff, anchor, mask_prototype = prediction

    box = box_decode(box_encoding, anchor, config)

    # TODO: Implement NMS for detections here

    for sample_i in range(img.size(0)):
        classification_max = torch.argmax(classification[sample_i], dim=-1).squeeze(0)
        detection = classification_max.nonzero().squeeze(-1)

        plot_prototype(mask_prototype[sample_i])

        mask = assemble_mask(mask_prototype[sample_i], mask_coeff[sample_i, detection], None)
        plot_mask(None, mask)
        plot_mask(img[sample_i], mask)

        plot_detection(
            img[sample_i],
            classification_max[detection],
            box[sample_i, detection],
            None,
            None,
            None
        )

    plt.show()

if __name__ == "__main__":
    main()
