"""
Dataloaders
"""

# Libs
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING
from sacred import Experiment

# Torch modules
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.distributed as dist

import cv2
import numpy as np
from config import initialise
from pathlib import Path

if TYPE_CHECKING:
    from utils.typing_alias import *


ex = Experiment("data")
ex = initialise(ex)


def resize(img, factor):
    num = int(-np.log2(factor))

    for i in range(num):
        dim_x = img.shape[0]
        dim_y = img.shape[1]
        pad_x = 1 if dim_x % 2 == 1 else 0
        pad_y = 1 if dim_y % 2 == 1 else 0
        img = 0.25 * (
            img[: dim_x - pad_x : 2, : dim_y - pad_y : 2]
            + img[1::2, : dim_y - pad_y : 2]
            + img[: dim_x - pad_x : 2, 1::2]
            + img[1::2, 1::2]
        )

    return img


def get_img_from_raw(raw, dataset_name: str = "phase_mask"):
    raw_h, raw_w = raw.shape

    if dataset_name == "phase_mask":
        img = np.zeros((raw_h // 2, raw_w // 2, 4))

        img[:, :, 0] = raw[0::2, 0::2]  # r
        img[:, :, 1] = raw[0::2, 1::2]  # gr
        img[:, :, 2] = raw[1::2, 0::2]  # gb
        img[:, :, 3] = raw[1::2, 1::2]  # b

        img = torch.tensor(img)

    elif dataset_name == "phase_mask_admm":
        img = np.zeros((raw_h // 2, raw_w // 2, 3))

        img[:, :, 0] = raw[0::2, 0::2]  # r
        img[:, :, 1] = 0.5 * (raw[0::2, 1::2] + raw[1::2, 0::2])  # g
        img[:, :, 2] = raw[1::2, 1::2]  # b
        img = torch.tensor(img)

    return img


@dataclass
class Data:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader = None


class PhaseMaskDataset(Dataset):
    """
    Assume folders have images one level deep
    """

    def __init__(
        self,
        args,
        mode: str = "train",
        max_len: int = None,
        is_local_rank_0: bool = True,
    ):
        super().__init__()

        assert mode in ["train", "val", "test"], "Mode can be train or val"
        self.mode = mode
        self.args = args
        self.image_dir = args.image_dir
        self.max_len = max_len

        self.source_paths, self.target_paths = self._load_dataset()

        if is_local_rank_0:
            logging.info(f"{mode.capitalize()} Set | Image Dir: {self.image_dir}")

    def _glob_images(self, file_list):
        with open(file_list) as f:
            source_paths = f.readlines()

        paths = [self.image_dir / Path(path.strip("\n")) for path in source_paths]
        return paths

    def _img_load(self, img_path: "Path" = None, img_mode="source", raw=[]):
        assert img_path or len(raw), "need either path or raw image"
        assert img_mode in ["source", "target"]
        if img_mode == "target":
            img = cv2.imread(str(img_path))[:, :, ::-1] / 255.0

            img = cv2.resize(img, (self.args.image_width, self.args.image_height))

        elif img_mode == "source":
            if not len(raw):
                raw = cv2.imread(str(img_path), -1)
            try:
                raw = raw / 4096.0
            except:
                breakpoint()

            img = get_img_from_raw(raw, self.args.dataset_name)

            # Crop
            if self.args.meas_crop_size_x and self.args.meas_crop_size_y:
                crop_x = self.args.meas_centre_x - self.args.meas_crop_size_x // 2
                crop_y = self.args.meas_centre_y - self.args.meas_crop_size_y // 2

                # Replicate padding
                img = img[
                    crop_x : crop_x + self.args.meas_crop_size_x,
                    crop_y : crop_y + self.args.meas_crop_size_y,
                ]

                pad_x = self.args.psf_height - self.args.meas_crop_size_x
                pad_y = self.args.psf_width - self.args.meas_crop_size_y

                img = F.pad(
                    img.permute(2, 0, 1).unsqueeze(0),
                    (pad_y // 2, pad_y // 2, pad_x // 2, pad_x // 2),
                    mode=self.args.pad_meas_mode,
                )

                img = img.squeeze(0).permute(1, 2, 0)

            if self.args.test_apply_gain and self.mode == "test":
                img = img / img.max() * (400 / 4096.0)

            if self.args.dataset_name == "phase_mask_admm":
                img = resize(img, 0.25)

        img = (img - 0.5) * 2  # Change range from -1,...,1
        img = np.transpose(img, (2, 0, 1))

        return img

    def _load_dataset(self):
        if self.mode == "train":
            source_paths = self._glob_images(self.args.train_source_list)[
                : self.max_len
            ]
            target_paths = self._glob_images(self.args.train_target_list)[
                : self.max_len
            ]

        elif self.mode == "val":
            source_paths = self._glob_images(self.args.val_source_list)[: self.max_len]
            target_paths = self._glob_images(self.args.val_target_list)[: self.max_len]

        elif self.mode == "test":
            source_paths = list(self.image_dir.glob(self.args.test_glob_pattern))
            target_paths = None

        return source_paths, target_paths

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        source_path = self.source_paths[index]
        source = self._img_load(source_path, img_mode="source")

        if self.mode == "test":
            return source.float(), f"{source_path.parent.name}/{source_path.name}"

        if self.mode == "train":
            source = source + torch.normal(
                torch.zeros_like(source), self.args.train_gaussian_noise
            )

        target_path = self.target_paths[index]
        target = self._img_load(target_path, img_mode="target")

        return (
            source.float(),
            torch.from_numpy(target.copy()).float(),
            source_path.name,
        )


def get_dataloaders(args, is_local_rank_0: bool = True):
    """
    Get dataloaders for train and val

    Returns:
    :data
    """

    train_dataset = PhaseMaskDataset(
        args, mode="train", is_local_rank_0=is_local_rank_0
    )
    val_dataset = PhaseMaskDataset(args, mode="val", is_local_rank_0=is_local_rank_0)
    test_dataset = PhaseMaskDataset(args, mode="test", is_local_rank_0=is_local_rank_0)

    if is_local_rank_0:
        logging.info(
            f"Dataset: {args.dataset_name} Len Train: {len(train_dataset)} Val: {len(val_dataset)}  Test: {len(test_dataset)}"
        )

    train_loader = None
    val_loader = None
    test_loader = None

    if len(train_dataset):
        if args.distdataparallel:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=dist.get_world_size(), shuffle=True
            )
            shuffle = False

        else:
            train_sampler = None
            shuffle = True

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_threads,
            pin_memory=False,
            drop_last=True,
            sampler=train_sampler,
        )

    if len(val_dataset):
        if args.distdataparallel:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, num_replicas=dist.get_world_size(), shuffle=True
            )
            shuffle = False

        else:
            val_sampler = None
            shuffle = True

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_threads,
            pin_memory=False,
            drop_last=True,
            sampler=val_sampler,
        )

    if len(test_dataset):
        if args.distdataparallel:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset, num_replicas=dist.get_world_size(), shuffle=True
            )
            shuffle = False

        else:
            test_sampler = None
            shuffle = True

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_threads,
            pin_memory=False,
            drop_last=True,
            sampler=test_sampler,
        )

    return Data(
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader
    )


@ex.automain
def main(_run):
    from tqdm import tqdm
    from utils.tupperware import tupperware

    args = tupperware(_run.config)

    data = get_dataloaders(args)

    for _ in tqdm(data.train_loader.dataset):
        pass
