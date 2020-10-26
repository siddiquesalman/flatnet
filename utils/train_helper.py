"""
Train Script for Unet

@MOD: 8th April 2019

@py3.6+

@requirements: See utils/requirments.txt
"""
# Libraries
from utils.model_serialization import load_state_dict

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.adamw import AdamW


# Torch Libs
import torch
import torch.distributed as dist

import logging

# Typing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing_alias import *


def reduce_loss_dict(loss_dict, world_size):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    if world_size < 2:
        return {k: v.item() for k, v in loss_dict.items()}
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v.item() for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def pprint_args(args):
    """
    Pretty print args
    """
    string = str(args)
    string_ll = string.replace("Tupperware(", "").rstrip(")").split(", ")
    string_ll = sorted(string_ll, key=lambda x: x.split("=")[0].lower())

    string_ll = [
        f"*{line.split('=')[0]}* = {line.split('=')[-1]}" for line in string_ll
    ]
    string = "\n".join(string_ll)

    return string


def get_optimisers(
    G: "nn.Module", FFT: "nn.Module", D: "nn.Module", args: "tupperware"
) -> "Tuple[optim, optim, lr_scheduler, lr_scheduler]":

    if len(args.G_finetune_layers):
        G_params = [
            {"params": getattr(G, attribute).parameters()}
            for attribute in args.G_finetune_layers
        ]

    else:
        G_params = G.parameters()

    g_optimizer = AdamW(
        G_params, lr=args.learning_rate, betas=(args.beta_1, args.beta_2)
    )
    fft_optimizer = AdamW(
        FFT.parameters(), lr=args.fft_learning_rate, betas=(args.beta_1, args.beta_2)
    )
    d_optimizer = AdamW(
        D.parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2)
    )

    if args.lr_scheduler == "cosine":
        g_lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer=g_optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=2e-7
        )

        fft_lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer=fft_optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=2e-15
        )

        d_lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer=d_optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=2e-7
        )
    elif args.lr_scheduler == "step":
        g_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=g_optimizer, step_size=args.step_size, gamma=0.1
        )

        fft_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=fft_optimizer, step_size=args.step_size, gamma=0.1
        )

        d_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=d_optimizer, step_size=args.step_size, gamma=0.1
        )

    return (
        (g_optimizer, fft_optimizer, d_optimizer),
        (g_lr_scheduler, fft_lr_scheduler, d_lr_scheduler),
    )


def load_models(
    G: "nn.Module" = None,
    FFT: "nn.Module" = None,
    D: "nn.Module" = None,
    g_optimizer: "optim" = None,
    fft_optimizer: "optim" = None,
    d_optimizer: "optim" = None,
    args: "tupperware" = None,
    tag: str = "latest",
    is_local_rank_0: bool = True,
) -> "Tuple[List[nn.module], List[optim], int, int, int]":

    names = ["Gen", "FFT", "Disc"]

    latest_paths = [
        (args.ckpt_dir / i).resolve()
        for i in [
            args.save_filename_latest_G,
            args.save_filename_latest_FFT,
            args.save_filename_latest_D,
        ]
    ]
    best_paths = [
        (args.ckpt_dir / i).resolve()
        for i in [args.save_filename_G, args.save_filename_FFT, args.save_filename_D]
    ]

    if tag == "latest":
        paths = latest_paths
        if not paths[0].exists():
            paths = best_paths
            tag = "best"

    elif tag == "best":
        paths = best_paths
        if not paths[0].exists():
            paths = latest_paths
            tag = "latest"

    models = [G, FFT, D]
    optimizers = [g_optimizer, fft_optimizer, d_optimizer]

    # Defaults
    start_epoch = 0
    global_step = 0
    loss = 1e6

    if args.resume:
        for name, path, model, optimizer in zip(names, paths, models, optimizers):
            if path.is_file():
                checkpoint = torch.load(path, map_location=torch.device("cpu"))

                if model:
                    load_state_dict(model, checkpoint["state_dict"])

                if not args.finetune:
                    if optimizer and "optimizer" in checkpoint:
                        optimizer.load_state_dict(checkpoint["optimizer"])

                    if name == "Gen":
                        if "epoch" in checkpoint:
                            start_epoch = checkpoint["epoch"] - 1

                        if "global_step" in checkpoint:
                            global_step = checkpoint["global_step"]

                        if "loss" in checkpoint:
                            loss = checkpoint["loss"]

                if is_local_rank_0:
                    logging.info(
                        f"Loading checkpoint for {name} from {path} with tag {tag} at epoch {start_epoch + 1} global step {global_step}."
                    )

            else:
                if is_local_rank_0:
                    logging.info(
                        f"No checkpoint found for {name} at {path} with tag {tag}."
                    )

    # Best model
    path = args.ckpt_dir / args.exp_name / args.save_filename_G
    if path.exists() and not args.finetune:
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        if "loss" in checkpoint:
            loss = checkpoint["loss"]

        if is_local_rank_0:
            logging.info(f"Previous best model has loss of {loss}")

    return models, optimizers, global_step, start_epoch, loss


def save_weights(
    global_step: int,
    epoch: int,
    G: "nn.Module" = None,
    FFT: "nn.Module" = None,
    D: "nn.Module" = None,
    g_optimizer: "optim" = None,
    fft_optimizer: "optim" = None,
    d_optimizer: "optim" = None,
    loss: "float" = None,
    is_min: bool = True,
    args: "tupperware" = None,
    tag: str = "latest",
    is_local_rank_0: bool = True,
):
    if is_min or tag == "latest":
        if is_local_rank_0:
            logging.info(f"Epoch {epoch + 1} saving weights")
        if G:
            # Gen
            G_state = {
                "global_step": global_step,
                "epoch": epoch + 1,
                "state_dict": G.state_dict(),
                "optimizer": g_optimizer.state_dict(),
                "loss": loss,
            }
            save_filename_G = (
                args.save_filename_latest_G if tag == "latest" else args.save_filename_G
            )

            path_G = str(args.ckpt_dir / save_filename_G)
            torch.save(G_state, path_G)

            # Specific saving
            if epoch % args.save_copy_every_epochs == 0 and tag == "latest":
                save_filename_G = f"Epoch_{epoch}_{save_filename_G}"

                path_G = str(args.ckpt_dir / save_filename_G)
                torch.save(G_state, path_G)

        if FFT:
            # FFT
            FFT_state = {
                "global_step": global_step,
                "epoch": epoch + 1,
                "state_dict": FFT.state_dict(),
                "optimizer": fft_optimizer.state_dict(),
                "loss": loss,
            }
            save_filename_FFT = (
                args.save_filename_latest_FFT
                if tag == "latest"
                else args.save_filename_FFT
            )

            path_FFT = str(args.ckpt_dir / save_filename_FFT)
            torch.save(FFT_state, path_FFT)

            if epoch % args.save_copy_every_epochs == 0 and tag == "latest":
                save_filename_FFT = f"Epoch_{epoch}_{save_filename_FFT}"

                path_FFT = str(args.ckpt_dir / save_filename_FFT)
                torch.save(FFT_state, path_FFT)

        if D:
            # Disc
            D_state = {
                "global_step": global_step,
                "epoch": epoch + 1,
                "state_dict": D.state_dict(),
                "optimizer": d_optimizer.state_dict(),
            }
            save_filename_D = (
                args.save_filename_latest_D if tag == "latest" else args.save_filename_D
            )

            path_D = str(args.ckpt_dir / save_filename_D)
            torch.save(D_state, path_D)

            if epoch % args.save_copy_every_epochs == 0 and tag == "latest":
                save_filename_D = f"Epoch_{epoch}_{save_filename_D}"

                path_D = str(args.ckpt_dir / save_filename_D)
                torch.save(D_state, path_D)
    else:
        if is_local_rank_0:
            logging.info(f"Epoch {epoch + 1} NOT saving weights")


class SmoothenValue(object):
    "Create a smooth moving average for a value (loss, etc) using `beta`."

    def __init__(self, beta: float = 0.9):
        self.beta, self.n, self.mov_avg = beta, 0, 0

    def add_value(self, val: float) -> None:
        "Add `val` to calculate updated smoothed value."
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * val
        self.smooth = self.mov_avg / (1 - self.beta ** self.n)


# Dictionary based loss collectors
# See train.py for usage
class AvgLoss_with_dict(object):

    """
    Utility class for storing running averages of losses
    """

    def __init__(self, loss_dict: "Dict", args: "tupperware", count: int = 0):
        self.args = args
        self.count = count
        self.loss_dict = loss_dict

    def reset(self):
        self.count = 0
        for k in self.loss_dict:
            self.loss_dict[k] = 0.0

    def __add__(self, loss_dict: "Dict"):
        self.count += 1

        assert loss_dict.keys() == self.loss_dict.keys(), "Keys donot match"

        for k in self.loss_dict:
            self.loss_dict[k] += (loss_dict[k] - self.loss_dict[k]) / self.count

        return self


class ExpLoss_with_dict(object):
    def __init__(self, loss_dict: "Dict", args: "tupperware"):
        """
        :param dict: Expects default dict
        """
        self.args = args
        self.loss_dict = loss_dict
        self.set_collector()

    def set_collector(self):
        self.collector_dict = {}
        for k in self.loss_dict:
            self.collector_dict[k + "_collector"] = SmoothenValue()

    def __add__(self, loss_dict: "Dict"):
        assert loss_dict.keys() == self.loss_dict.keys(), "Keys donot match"
        for k in self.loss_dict:
            self.collector_dict[k + "_collector"].add_value(loss_dict[k])
            self.loss_dict[k] = self.collector_dict[k + "_collector"].smooth

        return self
