import os
from typing import List, Dict

import torch
import wandb
from torch.utils.data import DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule

from ikflow.config import ALL_DATASET_TAGS, DATASET_DIR, device
from ikflow.utils import get_sum_joint_limit_range, get_dataset_filepaths


class IkfLitDataset(LightningDataModule):
    def __init__(
        self,
        robot_name: str,
        batch_size: int,
        val_set_size: int,
        dataset_tags: List[str],
        prepare_data_per_node: bool = True,
    ):
        """
        A LightningDataModule that not only provides train/val loaders
        but also exposes the exact (mean, std) normalizers used at training time
        via `._x_transform` and `._y_transform`.
        """
        # Save and sanity-check dataset tags
        self._dataset_tags = dataset_tags or []
        for tag in self._dataset_tags:
            assert tag in ALL_DATASET_TAGS, f"Unknown dataset tag: {tag}"

        self._robot_name = robot_name
        self._batch_size = batch_size
        self._val_set_size = val_set_size
        self.prepare_data_per_node = prepare_data_per_node
        self._log_hyperparams = True

        # Build the dataset directory path including tags
        suffix = "_" + "_".join(self._dataset_tags) if self._dataset_tags else ""
        dataset_directory = os.path.join(DATASET_DIR, f"{self._robot_name}{suffix}")
        assert os.path.isdir(dataset_directory), (
            f"Directory '{dataset_directory}' doesn't exist. "
            "Have you run the dataset-building script?"
        )

        # Locate .pt files
        (
            samples_tr_path,
            poses_tr_path,
            samples_te_path,
            poses_te_path,
            _,
        ) = get_dataset_filepaths(dataset_directory, self._dataset_tags)

        # Load tensors onto the correct device
        self._samples_tr = torch.load(samples_tr_path).to(device)
        self._endpoints_tr = torch.load(poses_tr_path).to(device)
        self._samples_te = torch.load(samples_te_path).to(device)
        self._endpoints_te = torch.load(poses_te_path).to(device)

        # Bookkeeping
        self._sum_joint_limit_range = get_sum_joint_limit_range(self._samples_tr)
        self.allow_zero_length_dataloader_with_multiple_devices = False

        # Compute normalization constants
        eps = 1e-6
        self._x_mean = self._samples_tr.mean(dim=0, keepdim=True)
        self._x_std = self._samples_tr.std(dim=0, keepdim=True).clamp_min(eps)
        self._y_mean = self._endpoints_tr.mean(dim=0, keepdim=True)
        self._y_std = self._endpoints_tr.std(dim=0, keepdim=True).clamp_min(eps)

        # Transforms for inputs/outputs
        self._x_transform = lambda x: (x - self._x_mean) / self._x_std
        self._y_transform = lambda y: (y - self._y_mean) / self._y_std

    def add_dataset_hashes_to_cfg(self, cfg: Dict):
        cfg.update({
            "dataset_hashes": str([
                self._samples_tr.sum().item(),
                self._endpoints_tr.sum().item(),
                self._samples_te.sum().item(),
                self._endpoints_te.sum().item(),
            ])
        })

    def log_dataset_sizes(self, epoch=0, batch_nb=0):
        """Log train/val sizes to wandb."""
        assert self._samples_tr.shape[0] == self._endpoints_tr.shape[0]
        assert self._samples_te.shape[0] == self._endpoints_te.shape[0]
        wandb.log({
            "epoch": epoch,
            "batch_nb": batch_nb,
            "sum_joint_limit_range": self._sum_joint_limit_range,
            "dataset_size_tr": self._samples_tr.shape[0],
            "dataset_size_te": self._samples_te.shape[0],
        })

    def train_dataloader(self):
        return DataLoader(
            torch.utils.data.TensorDataset(self._samples_tr, self._endpoints_tr),
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator(device=device),
        )

    def val_dataloader(self):
        return DataLoader(
            torch.utils.data.TensorDataset(
                self._samples_te[: self._val_set_size],
                self._endpoints_te[: self._val_set_size],
            ),
            batch_size=1,
            shuffle=False,
            drop_last=True,
        )
