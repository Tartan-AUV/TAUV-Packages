import torch
from math import pi
import pathlib
from torch.utils.data import DataLoader, ConcatDataset
import wandb
import albumentations as A
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from cloudpathlib import CloudPath, S3Client

from centernet.model.centernet import Centernet, initialize_weights, get_head_channels
from centernet.model.backbones.dla import DLABackbone
from utils.loss import loss
from centernet.model.config import ObjectConfig, ObjectConfigSet, AngleConfig, ModelConfig, TrainConfig
from centernet.model.backbones.centerpoint_dla import CenterpointDLA34
from centernet.configs.samples_torpedo_bin_buoy import model_config, train_config, object_config

from datasets.load.pose_dataset import PoseDataset, PoseSample, Split

torch.autograd.set_detect_anomaly(True)

client = S3Client(local_cache_dir="/home/gryabtse/data/tauv/s3-cache")

train_dataset_roots = [
    pathlib.Path("~/data/tauv/tauv-datasets/happen-right-city").expanduser(),# everything
    pathlib.Path("~/data/tauv/tauv-datasets/keep-happy-lot").expanduser(),  # torpedoes
]
val_dataset_roots = [
    pathlib.Path("~/data/tauv/tauv-datasets/happen-right-city").expanduser(),# everything
    pathlib.Path("~/data/tauv/tauv-datasets/keep-happy-lot").expanduser(),  # torpedoes
]
results_root = pathlib.Path("~/data/tauv/centernet_runs").expanduser()

# checkpoint_path = pathlib.Path("~/Documents/centernet_checkpoints/stellar-river-293_9.pt").expanduser()
checkpoint_path = None


def run_train_epoch(epoch_i: int, centernet: Centernet, optimizer, data_loader, train_config, device):
    centernet.train()

    for batch_i, batch in enumerate(data_loader):
        print(f"train epoch {epoch_i}, batch {batch_i}")

        optimizer.zero_grad()

        batch = batch.to(device)

        img = batch.img

        prediction = centernet(img)

        losses = loss(prediction, batch, model_config, train_config, object_config, img)

        total_loss = losses.total

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(centernet.parameters(), 1.0, norm_type=2.0, error_if_nonfinite=False,
                                       foreach=None)

        optimizer.step()

        wandb.log({"train_total_loss": losses.total})
        wandb.log({"train_heatmap_loss": losses.heatmap})
        wandb.log({"train_keypoint_heatmap_loss": losses.keypoint_heatmap})
        wandb.log({"train_keypoint_affinity_loss": losses.keypoint_affinity})
        wandb.log({"train_size_loss": losses.size})
        wandb.log({"train_offset_loss": losses.offset})
        wandb.log({"train_roll_loss": losses.roll})
        wandb.log({"train_pitch_loss": losses.pitch})
        wandb.log({"train_yaw_loss": losses.yaw})
        wandb.log({"train_depth_loss": losses.depth})

        wandb.log({"train_avg_size_error": losses.avg_size_error})
        wandb.log({"train_max_size_error": losses.max_size_error})


def run_validation_epoch(epoch_i, centernet, data_loader, device):
    centernet.eval()

    avg_losses = torch.zeros(10, dtype=torch.float32)
    n_batch = torch.zeros(1, dtype=torch.float)

    for batch_i, batch in enumerate(data_loader):
        print(f"val epoch {epoch_i}, batch {batch_i}")

        with torch.no_grad():
            batch = batch.to(device)

            img = batch.img

            prediction = centernet(img)

            if batch_i == 0:
                sample_i = 0

                heatmap = F.sigmoid(prediction.heatmap[batch_i, sample_i])
                fig, axs = plt.subplots()
                im = axs.imshow(heatmap.detach().cpu())
                fig.colorbar(im)
                wandb.log({f"val_heatmap_{batch_i}_{sample_i}": fig})
                plt.close(fig)

            losses = loss(prediction, batch, model_config, train_config, object_config, img)

            wandb.log({"val_total_loss": losses.total})
            wandb.log({"val_heatmap_loss": losses.heatmap})
            wandb.log({"val_keypoint_heatmap_loss": losses.keypoint_heatmap})
            wandb.log({"val_keypoint_affinity_loss": losses.keypoint_affinity})
            wandb.log({"val_size_loss": losses.size})
            wandb.log({"val_offset_loss": losses.offset})
            wandb.log({"val_roll_loss": losses.roll})
            wandb.log({"val_pitch_loss": losses.pitch})
            wandb.log({"val_yaw_loss": losses.yaw})
            wandb.log({"val_depth_loss": losses.depth})

            avg_losses += torch.Tensor((
                losses.total.cpu(),
                losses.heatmap.cpu(),
                losses.keypoint_heatmap.cpu(),
                losses.keypoint_affinity.cpu(),
                losses.size.cpu(),
                losses.offset.cpu(),
                losses.roll.cpu(),
                losses.pitch.cpu(),
                losses.yaw.cpu(),
                losses.depth.cpu(),
            ))
            n_batch += 1

    avg_losses /= n_batch

    wandb.log({"val_avg_total_loss": avg_losses[0]})
    wandb.log({"val_avg_heatmap_loss": avg_losses[1]})
    wandb.log({"val_avg_keypoint_heatmap_loss": avg_losses[2]})
    wandb.log({"val_avg_keypoint_affinity_loss": avg_losses[3]})
    wandb.log({"val_avg_size_loss": avg_losses[4]})
    wandb.log({"val_avg_offset_loss": avg_losses[5]})
    wandb.log({"val_avg_roll_loss": avg_losses[6]})
    wandb.log({"val_avg_pitch_loss": avg_losses[7]})
    wandb.log({"val_avg_yaw_loss": avg_losses[8]})
    wandb.log({"val_avg_depth_loss": avg_losses[9]})


train_transform = A.Compose(
    [
        A.ToGray(num_output_channels=1, method="weighted_average", p=1),
        A.RandomSizedCrop(
            min_max_height=(260, 360),
            w2h_ratio=640 / 360,
            width=640,
            height=360,
            p=0.25,
        ),
        A.Flip(),
        A.Blur(),
        A.GaussNoise(),
        A.PiecewiseAffine(scale=[0.01, 0.02], nb_rows=4, nb_cols=4, p=0.1),
        A.Normalize(mean=(0.5,), std=(0.5,), always_apply=True),
    ],
    bbox_params=A.BboxParams(format="albumentations", label_fields=["bbox_labels", "bbox_indices", "roll", "pitch", "yaw", "depth"]),
    keypoint_params=A.KeypointParams(format="xy", label_fields=["keypoint_labels", "keypoint_object_indices"]),
)


val_transform = A.Compose(
    [
        A.ToGray(num_output_channels=1, method="weighted_average", p=1),
        A.Normalize(mean=(0.5,), std=(0.5,), always_apply=True),
    ],
    bbox_params=A.BboxParams(format="albumentations",
                             label_fields=["bbox_labels", "bbox_indices", "roll", "pitch", "yaw", "depth"]),
    keypoint_params=A.KeypointParams(format="xy", label_fields=["keypoint_labels", "keypoint_object_indices"]),
)


def main():
    for checkpoint in results_root.iterdir():
        checkpoint.unlink()

    wandb.init(
        entity="tartanauv",
        project="centernet-grayscale",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")

    centernet = CenterpointDLA34(object_config, input_channels=1).to(device)
    if checkpoint_path is not None:
        centernet.load_state_dict(torch.load(checkpoint_path))
    centernet.train()

    optimizer = torch.optim.Adam(centernet.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1, total_iters=5),
        torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1),
        torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1),
    ], milestones=[5, 40])

    train_datasets = [
        PoseDataset(dataset_root, Split.TRAIN, object_config.label_id_to_index, object_config, train_transform)
        for dataset_root in train_dataset_roots
    ]
    train_dataset = ConcatDataset(train_datasets)
    val_datasets = [
        PoseDataset(dataset_root, Split.VAL, object_config.label_id_to_index, object_config, val_transform)
        for dataset_root in val_dataset_roots
    ]
    val_dataset = ConcatDataset(val_datasets)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        collate_fn=PoseSample.collate,
        shuffle=True,
        num_workers=train_config.n_workers,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        collate_fn=PoseSample.collate,
        shuffle=True,
        num_workers=train_config.n_workers,
    )

    for epoch_i in range(train_config.n_epochs):
        save_path = results_root / f"latest.pt"
        torch.save(centernet.state_dict(), save_path)

        if epoch_i % train_config.weight_save_interval == 0:
            save_path = results_root / f"{epoch_i}.pt"
            torch.save(centernet.state_dict(), save_path)
            artifact = wandb.Artifact('model', type='model')
            artifact.add_dir(results_root)
            wandb.log_artifact(artifact)

        run_train_epoch(epoch_i, centernet, optimizer, train_dataloader, train_config, device)

        run_validation_epoch(epoch_i, centernet, val_dataloader, device)

        wandb.log({ 'lr': scheduler.get_last_lr() })

        scheduler.step()

    save_path = results_root / f"{epoch_i}.pt"
    torch.save(centernet.state_dict(), save_path)
    artifact = wandb.Artifact('model', type='model')
    artifact.add_dir(results_root)
    wandb.log_artifact(artifact)


if __name__ == "__main__":
    main()