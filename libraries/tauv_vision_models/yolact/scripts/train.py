import cv2
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
from typing import List, Tuple, Optional
from pathlib import Path
import pathlib
from dataclasses import asdict
import wandb
import matplotlib.pyplot as plt
import albumentations as A

from tauv_vision.yolact.model.config import ModelConfig, TrainConfig, ClassConfig, ClassConfigSet
from tauv_vision.yolact.model.loss import loss
from tauv_vision.yolact.model.model import Yolact
from tauv_vision.yolact.model.weights import initialize_weights
from tauv_vision.yolact.model.boxes import box_decode, box_to_corners, corners_to_box
from tauv_vision.yolact.model.masks import assemble_mask
from tauv_vision.datasets.segmentation_dataset.segmentation_dataset import SegmentationDataset, SegmentationSample, \
    SegmentationDatasetSet
from tauv_vision.utils.plot import save_plot, plot_prototype, plot_mask, plot_detection

# TODO: Set train script up to take in a .json class list
# It'll need to pass the class list to every instantiation of SegmentationDataset
# so the dataset can combine it with the dataset class list to set the class ids properly


model_config = ModelConfig(
    in_w=640,
    in_h=360,
    feature_depth=256,
    n_classes=7,
    n_prototype_masks=8,
    n_masknet_layers_pre_upsample=1,
    n_masknet_layers_post_upsample=1,
    n_prediction_head_layers=1,
    n_classification_layers=0,
    n_box_layers=0,
    n_mask_layers=0,
    n_fpn_downsample_layers=2,
    anchor_scales=(24, 48, 96, 192, 384),
    anchor_aspect_ratios=(1,),
    box_variances=(0.1, 0.2),
    iou_pos_threshold=0.4,
    iou_neg_threshold=0.3,
    negative_example_ratio=3,
    img_mean=(0.485, 0.456, 0.406),
    img_stddev=(0.229, 0.224, 0.225),
)

train_config = TrainConfig(
    lr=1e-3,
    momentum=0.9,
    weight_decay=0,
    grad_max_norm=1e0,
    n_epochs=200,
    batch_size=24,
    epoch_n_batches=100,
    weight_save_interval=1,
    gradient_save_frequency=1000,
    channel_shuffle_p=0,
    color_jitter_p=1,
    color_jitter_brightness=0.2,
    color_jitter_contrast=0.2,
    color_jitter_saturation=0.2,
    color_jitter_hue=0.2,
    gaussian_noise_p=1.0,
    gaussian_noise_var_limit=(10.0, 50.0),
    horizontal_flip_p=0.5,
    vertical_flip_p=0.5,
    blur_limit=(3, 7),
    blur_p=0.5,
    ssr_p=1,
    ssr_shift_limit=(-0.1, 0.1),
    ssr_scale_limit=(-0.1, 0.1),
    ssr_rotate_limit=(-30, 30),
    perspective_p=1,
    perspective_scale_limit=(0.0, 0.25),
    min_visibility=0.0,
    n_workers=4,
)

class_config = ClassConfigSet([
    ClassConfig(
        id="torpedo_22_circle",
        index=1,
    ),
    ClassConfig(
        id="torpedo_22_trapezoid",
        index=2,
    ),
    ClassConfig(
        id="torpedo_22_star",
        index=3,
    ),
    ClassConfig(
        id="buoy_23_abydos_1",
        index=4,
    ),
    ClassConfig(
        id="buoy_23_abydos_2",
        index=5,
    ),
    ClassConfig(
        id="buoy_23_earth_1",
        index=6,
    ),
    ClassConfig(
        id="buoy_23_earth_2",
        index=7,
    ),
])

train_dataset_roots = [
    # pathlib.Path("~/Documents/2023-11-05").expanduser(),
    # pathlib.Path("~/Documents/torpedo_22_2_small").expanduser(),
    pathlib.Path("~/Documents/TAUV-Datasets/watch-open-reason").expanduser(),
]
val_dataset_root = pathlib.Path("~/Documents/TAUV-Datasets/watch-open-reason").expanduser()
results_root = pathlib.Path("~/Documents/yolact_runs").expanduser()


def collate_samples(samples: List[SegmentationSample]) -> SegmentationSample:
    n_detections = [sample.valid.size(0) for sample in samples]
    max_n_detections = max(n_detections)

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
    img = torch.stack([sample.img for sample in samples], dim=0)
    seg = torch.stack([sample.seg for sample in samples], dim=0)
    img_valid = torch.stack([sample.img_valid for sample in samples], dim=0)

    corners = box_to_corners(bounding_boxes)
    corners = torch.clamp(corners, min=0, max=1)
    bounding_boxes = corners_to_box(corners)

    sample = SegmentationSample(
        img=img,
        seg=seg,
        valid=valid,
        classifications=classifications,
        bounding_boxes=bounding_boxes,
        img_valid=img_valid,
    )

    return sample


def prepare_batch(batch: SegmentationSample, device: torch.device) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    img = batch.img.to(device)

    truth = (
        batch.valid.to(device),
        batch.classifications.to(device),
        batch.bounding_boxes.to(device),
        batch.seg.to(device),
        batch.img_valid.to(device),
    )

    return img, truth


def plot_train_batch(epoch_i: int, batch_i: int, img: torch.Tensor, prediction: Tuple[torch.Tensor, ...],
                     truth: Tuple[torch.Tensor, ...], model_config: ModelConfig, save_dir: Optional[pathlib.Path] = None):
    classification, box_encoding, mask_coeff, anchor, mask_prototype = prediction
    truth_valid, truth_classification, truth_box, truth_seg_map, truth_img_valid = truth

    sample_i = 0

    classification_max = torch.argmax(classification[sample_i], dim=-1).squeeze(0)
    detections = classification_max.nonzero().squeeze(-1)[:100]

    prototype_fig = plot_prototype(mask_prototype[sample_i])
    prototype_fig.suptitle(f"Epoch {epoch_i} Batch {batch_i} Sample {sample_i} Prototypes")
    prototype_fig.set_size_inches(16, 10)
    wandb.log({f"train_prototype_{batch_i}_{sample_i}": prototype_fig})
    save_plot(prototype_fig, save_dir, f"train_prototype_{epoch_i}_{batch_i}_{sample_i}")

    if len(detections) > 0:
        mask = assemble_mask(mask_prototype[sample_i], mask_coeff[sample_i, detections], box=None)
        mask_fig = plot_mask(None, mask)
        mask_fig.suptitle(f"Epoch {epoch_i} Batch {batch_i} Sample {sample_i} Masks")
        mask_fig.set_size_inches(16, 10)
        wandb.log({f"train_mask_{batch_i}_{sample_i}": mask_fig})
        save_plot(mask_fig, save_dir, f"train_mask_{epoch_i}_{batch_i}_{sample_i}")

        mask_overlay_fig = plot_mask(img[sample_i], mask)
        mask_overlay_fig.suptitle(f"Epoch {epoch_i} Batch {batch_i} Sample {sample_i} Mask Overlays")
        mask_overlay_fig.set_size_inches(16, 10)
        wandb.log({f"train_mask_overlay_{batch_i}_{sample_i}": mask_overlay_fig})
        save_plot(mask_overlay_fig, save_dir, f"train_mask_overlay_{epoch_i}_{batch_i}_{sample_i}")

        box = box_decode(box_encoding, anchor, model_config)
        detection_fig = plot_detection(
            img[sample_i],
            classification_max[detections],
            box[sample_i, detections],
            truth_valid[sample_i],
            truth_classification[sample_i],
            truth_box[sample_i],
        )
        detection_fig.suptitle(f"Epoch {epoch_i} Batch {batch_i} Sample {sample_i} Detections")
        wandb.log({f"train_detection_{batch_i}_{sample_i}": detection_fig})
        save_plot(detection_fig, save_dir, f"train_detection_{epoch_i}_{batch_i}_{sample_i}")

        plt.close("all")


# From https://stackoverflow.com/questions/47714643/pytorch-data-loader-multiple-iterations
def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def run_train_epoch(epoch_i: int, model: Yolact, optimizer: torch.optim.Optimizer,
                    data_loader: DataLoader, train_config: TrainConfig, device: torch.device):
    model.train()

    data_loader_cycle = iter(cycle(data_loader))

    for batch_i, batch in enumerate(data_loader_cycle):
        if batch_i >= train_config.epoch_n_batches:
            break

        print(f"train epoch {epoch_i}, batch {batch_i}")
        img, truth = prepare_batch(batch, device=device)

        optimizer.zero_grad()

        prediction = model(img)

        if batch_i == train_config.epoch_n_batches - 1:
            plot_train_batch(epoch_i, batch_i, img, prediction, truth, model.config)

        total_loss, (classification_loss, box_loss, mask_loss) = loss(prediction, truth, model_config)

        print(f"total loss: {float(total_loss)}")
        print(
            f"classification loss: {float(classification_loss)}, box loss: {float(box_loss)}, mask loss: {float(mask_loss)}")

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_max_norm, norm_type=2.0, error_if_nonfinite=False,
                                       foreach=None)

        optimizer.step()

        wandb.log({"train_total_loss": total_loss})
        wandb.log({"train_classification_loss": classification_loss})
        wandb.log({"train_box_loss": box_loss})
        wandb.log({"train_mask_loss": mask_loss})


def plot_validation_batch(epoch_i: int, batch_i: int, img: torch.Tensor, prediction: Tuple[torch.Tensor, ...],
                          truth: Tuple[torch.Tensor, ...], model_config: ModelConfig, save_dir: Optional[pathlib.Path] = None):
    classification, box_encoding, mask_coeff, anchor, mask_prototype = prediction
    truth_valid, truth_classification, truth_box, truth_seg_map, truth_img_valid = truth

    n_batch = img.size(0)

    for sample_i in range(min(n_batch, 4)):
        classification_max = torch.argmax(classification[sample_i], dim=-1).squeeze(0)
        detections = classification_max.nonzero().squeeze(-1)[:100]

        prototype_fig = plot_prototype(mask_prototype[sample_i])
        prototype_fig.suptitle(f"Epoch {epoch_i} Batch {batch_i} Sample {sample_i} Prototypes")
        prototype_fig.set_size_inches(16, 10)
        wandb.log({f"val_prototype_{batch_i}_{sample_i}": prototype_fig})
        save_plot(prototype_fig, save_dir, f"val_prototype_{epoch_i}_{batch_i}_{sample_i}")

        if len(detections) > 0:
            mask = assemble_mask(mask_prototype[sample_i], mask_coeff[sample_i, detections], box=None)
            mask_fig = plot_mask(None, mask)
            mask_fig.suptitle(f"Epoch {epoch_i} Batch {batch_i} Sample {sample_i} Masks")
            mask_fig.set_size_inches(16, 10)
            wandb.log({f"val_mask_{batch_i}_{sample_i}": mask_fig})
            save_plot(mask_fig, save_dir, f"val_mask_{epoch_i}_{batch_i}_{sample_i}")

            mask_overlay_fig = plot_mask(img[sample_i], mask)
            mask_overlay_fig.suptitle(f"Epoch {epoch_i} Batch {batch_i} Sample {sample_i} Mask Overlays")
            mask_overlay_fig.set_size_inches(16, 10)
            wandb.log({f"val_mask_overlay_{batch_i}_{sample_i}": mask_overlay_fig})
            save_plot(mask_overlay_fig, save_dir, f"val_mask_overlay_{epoch_i}_{batch_i}_{sample_i}")

            box = box_decode(box_encoding, anchor, model_config)
            detection_fig = plot_detection(
                img[sample_i],
                classification_max[detections],
                box[sample_i, detections],
                truth_valid[sample_i],
                truth_classification[sample_i],
                truth_box[sample_i],
            )
            detection_fig.suptitle(f"Epoch {epoch_i} Batch {batch_i} Sample {sample_i} Detections")
            wandb.log({f"val_detection_{batch_i}_{sample_i}": detection_fig})
            save_plot(detection_fig, save_dir, f"val_detection_{epoch_i}_{batch_i}_{sample_i}")

            plt.close("all")

        plt.close("all")


def run_validation_epoch(epoch_i: int, model: Yolact, data_loader: DataLoader, train_config: TrainConfig, device: torch.device) -> float:
    model.eval()

    avg_losses = torch.zeros(4, dtype=torch.float)
    n_batch = torch.zeros(1, dtype=torch.float)

    for batch_i, batch in enumerate(data_loader):
        print(f"val epoch {epoch_i}, batch {batch_i}")

        with torch.no_grad():
            img, truth = prepare_batch(batch, device=device)

            prediction = model.forward(img)

            if batch_i == 0 and epoch_i > 0:
                plot_validation_batch(epoch_i, batch_i, img, prediction, truth, model.config)

            total_loss, (classification_loss, box_loss, mask_loss) = loss(prediction, truth, model.config)

            wandb.log({"val_total_loss": total_loss})
            wandb.log({"val_classification_loss": classification_loss})
            wandb.log({"val_box_loss": box_loss})
            wandb.log({"val_mask_loss": mask_loss})

        print(f"total loss: {float(total_loss)}")
        print(
            f"classification loss: {float(classification_loss)}, box loss: {float(box_loss)}, mask loss: {float(mask_loss)}")

        avg_losses += torch.Tensor((total_loss, classification_loss, box_loss, mask_loss))
        n_batch += 1

    avg_losses /= n_batch
    avg_total_loss, avg_classification_loss, avg_box_loss, avg_mask_loss = avg_losses

    print("validation averages:")
    print(f"total loss: {float(avg_total_loss)}")
    print(
        f"classification loss: {float(avg_classification_loss)}, box loss: {float(avg_box_loss)}, mask loss: {float(avg_mask_loss)}")

    wandb.log({"val_avg_total_loss": avg_total_loss})
    wandb.log({"val_avg_classification_loss": avg_classification_loss})
    wandb.log({"val_avg_box_loss": avg_box_loss})
    wandb.log({"val_avg_mask_loss": avg_mask_loss})

    return float(avg_total_loss)


def save_model(model: Yolact, save_dir: pathlib.Path, name: str):
    save_path = save_dir / f"{name}.pt"
    torch.save(model.state_dict(), save_path)
    artifact = wandb.Artifact(name, type='model')
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)


def main():
    save_dir = Path(results_root).expanduser()
    for checkpoint in save_dir.iterdir():
        checkpoint.unlink()

    run = wandb.init(
        project="yolact",
        config={
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "class_config": asdict(class_config),
            # TODO: Log datasets being trained on
        },
    )

    class_ids_to_indices = {c.id: c.index for c in class_config.configs}

    model_config_path = save_dir / f"{run.name}_model_config.json"
    train_config_path = save_dir / f"{run.name}_train_config.json"
    class_config_path = save_dir / f"{run.name}_class_config.json"

    model_config.save(model_config_path)
    train_config.save(train_config_path)
    class_config.save(class_config_path)

    model_config_artifact = wandb.Artifact(name=f"{run.name}_model_config", type="model_config")
    model_config_artifact.add_file(model_config_path)
    wandb.log_artifact(model_config_artifact)

    train_config_artifact = wandb.Artifact(name=f"{run.name}_train_config", type="train_config")
    train_config_artifact.add_file(train_config_path)
    wandb.log_artifact(train_config_artifact)

    class_config_artifact = wandb.Artifact(name=f"{run.name}_class_config", type="class_config")
    class_config_artifact.add_file(class_config_path)
    wandb.log_artifact(class_config_artifact)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Yolact(model_config).to(device)
    initialize_weights(model, [model._backbone])

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)

    train_transform = A.Compose(
        [
            A.ChannelShuffle(p=train_config.channel_shuffle_p),
            A.Resize(
                height=model_config.in_h, width=model_config.in_w, always_apply=True
            ),
            A.ColorJitter(
                brightness=train_config.color_jitter_brightness,
                contrast=train_config.color_jitter_contrast,
                saturation=train_config.color_jitter_saturation,
                hue=train_config.color_jitter_hue,
                p=train_config.color_jitter_p,
            ),
            A.GaussNoise(
                var_limit=train_config.gaussian_noise_var_limit,
                p=train_config.gaussian_noise_p,
            ),
            A.HorizontalFlip(p=train_config.horizontal_flip_p),
            A.VerticalFlip(p=train_config.vertical_flip_p),
            A.Blur(
                blur_limit=train_config.blur_limit,
                p=train_config.blur_p,
            ),
            A.ShiftScaleRotate(
                shift_limit=train_config.ssr_shift_limit,
                scale_limit=train_config.ssr_scale_limit,
                rotate_limit=train_config.ssr_rotate_limit,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=254,
                p=train_config.ssr_p,
            ),
            A.Perspective(
                scale=train_config.perspective_scale_limit,
                pad_mode=cv2.BORDER_CONSTANT,
                pad_val=0,
                mask_pad_val=254,
                p=train_config.perspective_p,
            ),
            A.Normalize(mean=model_config.img_mean, std=model_config.img_stddev, always_apply=True),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["classifications"], min_visibility=train_config.min_visibility),
    )

    val_transform = A.Compose(
        [
            A.Resize(height=model_config.in_h, width=model_config.in_w),
            A.Normalize(mean=model_config.img_mean, std=model_config.img_stddev),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["classifications"]),
    )

    train_datasets = [
        SegmentationDataset(dataset_root, SegmentationDatasetSet.TRAIN, class_ids_to_indices, transform=train_transform)
        for dataset_root in train_dataset_roots
    ]
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = SegmentationDataset(val_dataset_root, SegmentationDatasetSet.VALIDATION, class_ids_to_indices, transform=val_transform)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        collate_fn=collate_samples,
        shuffle=True,
        num_workers=train_config.n_workers,
    )

    wandb.watch(model, log="all", log_freq=train_config.gradient_save_frequency)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        collate_fn=collate_samples,
        shuffle=False,
        num_workers=train_config.n_workers,
    )

    best_val_loss = float("inf")

    epoch_i = 0
    for epoch_i in range(train_config.n_epochs):

        run_train_epoch(epoch_i, model, optimizer, train_dataloader, train_config, device)

        val_loss = run_validation_epoch(epoch_i, model, val_dataloader, train_config, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, save_dir, f"{run.name}_{epoch_i}_best.pt")
        elif epoch_i % train_config.weight_save_interval == 0:
            save_model(model, save_dir, f"{run.name}_{epoch_i}.pt")

    save_model(model, save_dir, f"{run.name}_{epoch_i}.pt")

if __name__ == "__main__":
    main()
