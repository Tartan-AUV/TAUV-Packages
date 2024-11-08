import argparse
import pathlib
import glob
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import cv2

from tauv_vision.yolact.model.config import ModelConfig
from tauv_vision.yolact.model.boxes import box_decode
from tauv_vision.yolact.model.nms import nms
from tauv_vision.yolact.model.model import Yolact
from tauv_vision.yolact.model.masks import assemble_mask


# ffmpeg -f image2 -r 25 -pattern_type glob -i '*.png' -vcodec libx264 -crf 22 video.mp4


config = ModelConfig(
    # in_w=1280,
    # in_h=720,
    in_w=640,
    in_h=360,
    feature_depth=64,
    n_classes=2,
    n_prototype_masks=16,
    n_masknet_layers_pre_upsample=1,
    n_masknet_layers_post_upsample=1,
    n_prediction_head_layers=1,
    n_classification_layers=0,
    n_box_layers=0,
    n_mask_layers=0,
    n_fpn_downsample_layers=2,
    anchor_scales=(24, 48, 96, 192, 384), # TODO: Check this. Like really check it.
    # anchor_aspect_ratios=(1 / 2, 1, 2),
    anchor_aspect_ratios=(1,),
    iou_pos_threshold=0.4,
    iou_neg_threshold=0.3,
    negative_example_ratio=3,
)

img_mean = (0.485, 0.456, 0.406)
img_stddev = (0.229, 0.224, 0.225)

top_k = 100
iou_threshold = 0.5
confidence_threshold = 0.5

def run(in_dir: pathlib.Path, out_dir: pathlib.Path, weights_path: pathlib.Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Yolact(config).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    img_names = glob.glob("*.png", root_dir=in_dir)
    img_paths = [in_dir / name for name in img_names]

    val_transform = A.Compose(
        [
            A.Resize(height=360, width=640),
            A.Normalize(mean=img_mean, std=img_stddev),
            ToTensorV2(),
        ],
    )

    for img_path in tqdm(img_paths):
        out_path = out_dir / img_path.name

        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)

        img = val_transform(image=img_np)["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model.forward(img)

        classification, box_encoding, mask_coeff, anchor, mask_prototype = prediction

        box = box_decode(box_encoding, anchor, config)

        detection = nms(classification, box, top_k, iou_threshold, confidence_threshold)

        box_detection = box[0, detection]
        classification_detection = classification[0, detection]
        confidence_detection = F.softmax(classification_detection, dim=-1)

        label_detection = torch.argmax(confidence_detection, dim=-1)

        n_detections = label_detection.size(0)

        if n_detections > 0:
            mask_coeff_detection = mask_coeff[0, detection]
            mask = assemble_mask(mask_prototype[0], mask_coeff_detection, box_detection)
            mask = F.interpolate(mask.unsqueeze(0), (config.in_h, config.in_w), mode="bilinear").squeeze(0)
            mask = mask > 0.5

        vis_img_np = cv2.resize(img_np, (config.in_w, config.in_h))

        cmap = matplotlib.colormaps.get_cmap("tab10")

        for detection_i in range(n_detections):
            color = cmap(int(label_detection[detection_i]))
            color = (255 * np.array(color)[:3])
            color = (int(color[0]), int(color[1]), int(color[2]))

            img_h = vis_img_np.shape[0]
            img_w = vis_img_np.shape[1]

            x0y0 = (
                int(img_w * (box_detection[detection_i, 1] - box_detection[detection_i, 3] / 2)),
                int(img_h * (box_detection[detection_i, 0] - box_detection[detection_i, 2] / 2)),
            )

            x1y1 = (
                int(img_w * (box_detection[detection_i, 1] + box_detection[detection_i, 3] / 2)),
                int(img_h * (box_detection[detection_i, 0] + box_detection[detection_i, 2] / 2)),
            )

            vis_img_np = cv2.rectangle(vis_img_np, x0y0, x1y1, color, 2)

            confidence_str = f"{round(float(confidence_detection[detection_i, label_detection[detection_i]]), 2)}"
            if x0y0[1] > 20:
                confidence_text_pos = (x0y0[0], x0y0[1] - 10)
            else:
                confidence_text_pos = (x0y0[0], x1y1[1] + 30)

            vis_img_np = cv2.putText(vis_img_np, confidence_str, confidence_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            alpha = 0.5

            mask_np = mask[detection_i].cpu().numpy()
            vis_img_np[mask_np] = alpha * np.array(color) + (1 - alpha) * (vis_img_np[mask_np])

        vis_img_pil = Image.fromarray(vis_img_np)
        vis_img_pil.save(out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir")
    parser.add_argument("out_dir")
    parser.add_argument("weights_path")

    args = parser.parse_args()

    in_dir = pathlib.Path(args.in_dir).expanduser()
    out_dir = pathlib.Path(args.out_dir).expanduser()
    weights_path = pathlib.Path(args.weights_path).expanduser()

    run(in_dir, out_dir, weights_path)


if __name__ == "__main__":
    main()