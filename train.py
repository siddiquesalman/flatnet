"""
Train Script for Phase Mask and Amplitude Mask
"""
# Libraries
from sacred import Experiment
from tqdm import tqdm
from collections import defaultdict
import logging
import numpy as np
import os, sys, warnings

# Torch Libs
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

# Modules
from dataloader import get_dataloaders
from utils.dir_helper import dir_init
from models import get_model
from loss import GLoss, DLoss
from config import initialise
from metrics import PSNR

# Typing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing_alias import *

# Train helpers
from utils.train_helper import (
    reduce_loss_dict,
    get_optimisers,
    load_models,
    save_weights,
    ExpLoss_with_dict,
    AvgLoss_with_dict,
    pprint_args,
)
from utils.ops import rggb_2_rgb, unpixel_shuffle
from utils.tupperware import tupperware

# Experiment, add any observers by command line
ex = Experiment("Train")
ex = initialise(ex)

# local rank 0: for logging, saving ckpts
if "LOCAL_RANK" in os.environ:
    is_local_rank_0 = int(os.environ["LOCAL_RANK"]) == 0
else:
    is_local_rank_0 = True
if not is_local_rank_0:
    sys.stdout = open(os.devnull, "w")

# To prevent "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy("file_system")
torch.autograd.set_detect_anomaly(True)


@ex.automain
def main(_run):
    args = tupperware(_run.config)

    # Dir init
    dir_init(args, is_local_rank_0=is_local_rank_0)

    # Ignore warnings
    if not is_local_rank_0:
        warnings.filterwarnings("ignore")

    # Mutli GPUS Setup
    if args.distdataparallel:
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
    else:
        rank = args.device
        world_size = 1

    # Is ADMM
    is_admm = "admm" in args.exp_name
    interm_name = "FFT" if not is_admm else "ADMM"

    # Get data
    data = get_dataloaders(args, is_local_rank_0=is_local_rank_0)

    # Model
    G, FFT, D = get_model.model(args)
    G = G.to(rank)
    FFT = FFT.to(rank)
    D = D.to(rank)

    # Optimisers
    (g_optimizer, fft_optimizer, d_optimizer), (
        g_lr_scheduler,
        fft_lr_scheduler,
        d_lr_scheduler,
    ) = get_optimisers(G, FFT, D, args)

    # Load Models
    (G, FFT, D), (
        g_optimizer,
        fft_optimizer,
        d_optimizer,
    ), global_step, start_epoch, loss = load_models(
        G,
        FFT,
        D,
        g_optimizer,
        fft_optimizer,
        d_optimizer,
        args,
        is_local_rank_0=is_local_rank_0,
    )

    if args.distdataparallel:
        # Wrap with Distributed Data Parallel
        G = torch.nn.parallel.DistributedDataParallel(
            G, device_ids=[rank], output_device=rank
        )
        FFT = torch.nn.parallel.DistributedDataParallel(
            FFT, device_ids=[rank], output_device=rank
        )
        D = torch.nn.parallel.DistributedDataParallel(
            D, device_ids=[rank], output_device=rank
        )

    writer = SummaryWriter(log_dir=str(args.run_dir))
    writer.add_text("Args", pprint_args(args))

    # Log no of GPUs
    if is_local_rank_0:
        world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        logging.info("Using {} GPUs".format(world_size))
        writer.add_text("Args", pprint_args(args))

        # Pbars
        train_pbar = tqdm(
            range(len(data.train_loader) * args.batch_size), dynamic_ncols=True
        )

        val_pbar = (
            tqdm(range(len(data.val_loader) * args.batch_size), dynamic_ncols=True)
            if data.val_loader
            else None
        )

        test_pbar = (
            tqdm(range(len(data.test_loader) * args.batch_size), dynamic_ncols=True)
            if data.test_loader
            else None
        )

    # Initialise losses
    g_loss = GLoss(args).to(rank)
    d_loss = DLoss(args).to(rank)

    # Compatibility with checkpoints without global_step
    if not global_step:
        global_step = start_epoch * len(data.train_loader) * args.batch_size

    start_epoch = global_step // len(data.train_loader.dataset)

    # Exponential averaging of loss
    loss_dict = {
        "g_loss": 0.0,
        "d_loss": 0.0,
        "adversarial_loss": 0.0,
        "perception_loss": 0.0,
        "contextual_loss": 0.0,
        "image_loss": 0.0,
        "train_PSNR": 0.0,
    }
    if not is_admm:
        loss_dict["fft_grad_norm"] = 0.0

    metric_dict = {"PSNR": 0.0, "g_loss": 0.0}
    avg_metrics = AvgLoss_with_dict(loss_dict=metric_dict, args=args)
    exp_loss = ExpLoss_with_dict(loss_dict=loss_dict, args=args)

    try:

        for epoch in range(start_epoch, args.num_epochs):
            # Train mode
            G.train()
            FFT.train()
            D.train()

            if is_local_rank_0:
                train_pbar.reset()

            if args.distdataparallel:
                data.train_loader.sampler.set_epoch(epoch)

            for i, batch in enumerate(data.train_loader):
                # allows for interrupted training
                if (
                    (global_step + 1) % (len(data.train_loader) * args.batch_size) == 0
                ) and (epoch == start_epoch):
                    break

                loss_dict = defaultdict(float)

                source, target, filename = batch
                source, target = (source.to(rank), target.to(rank))

                n, c, h, w = target.shape
                target_unpixel_shuffled = unpixel_shuffle(
                    target, args.pixelshuffle_ratio
                ).reshape(
                    n * args.pixelshuffle_ratio ** 2,
                    c,
                    h // args.pixelshuffle_ratio,
                    w // args.pixelshuffle_ratio,
                )

                # ------------------------------- #
                # Update Disc
                # ------------------------------- #
                D.zero_grad()
                G.zero_grad()
                FFT.zero_grad()

                fft_output = FFT(source)

                if is_admm:
                    # Upsample
                    fft_output = F.interpolate(
                        fft_output, scale_factor=4, mode="nearest"
                    )

                # Unpixelshuffle
                fft_unpixel_shuffled = unpixel_shuffle(
                    fft_output, args.pixelshuffle_ratio
                )

                output_unpixel_shuffled = G(fft_unpixel_shuffled)

                n, c, h, w = output_unpixel_shuffled.shape
                output_unpixel_shuffled = output_unpixel_shuffled.reshape(
                    n * args.pixelshuffle_ratio ** 2,
                    c // args.pixelshuffle_ratio ** 2,
                    h,
                    w,
                ).detach()

                real_logit = D(target_unpixel_shuffled)
                fake_logit = D(output_unpixel_shuffled)

                d_loss(real_logit=real_logit, fake_logit=fake_logit)

                d_loss.total_loss.backward()

                loss_dict["d_loss"] += d_loss.total_loss
                d_optimizer.step()

                # ------------------------------- #
                # Update Gen
                # ------------------------------- #
                G.zero_grad()
                FFT.zero_grad()

                fft_output = FFT(source)

                if is_admm:
                    # Upsample
                    fft_output = F.interpolate(
                        fft_output, scale_factor=4, mode="nearest"
                    )

                # Unpixelshuffle
                fft_unpixel_shuffled = unpixel_shuffle(
                    fft_output, args.pixelshuffle_ratio
                )
                output_unpixel_shuffled = G(fft_unpixel_shuffled)

                n, c, h, w = output_unpixel_shuffled.shape
                output_unpixel_shuffled = output_unpixel_shuffled.reshape(
                    n * args.pixelshuffle_ratio ** 2,
                    c // args.pixelshuffle_ratio ** 2,
                    h,
                    w,
                )

                fake_logit = D(output_unpixel_shuffled)

                g_loss(
                    output=output_unpixel_shuffled,
                    target=target_unpixel_shuffled,
                    fake_logit=fake_logit,
                )
                g_loss.total_loss.backward()

                n, c, h, w = output_unpixel_shuffled.shape
                output_unpixel_shuffled = output_unpixel_shuffled.reshape(
                    n // args.pixelshuffle_ratio ** 2,
                    c * args.pixelshuffle_ratio ** 2,
                    h,
                    w,
                )
                output = F.pixel_shuffle(
                    output_unpixel_shuffled, args.pixelshuffle_ratio
                )

                # Train PSNR
                loss_dict["train_PSNR"] += PSNR(output, target)

                # Gradient norm of FFT Layer
                if not is_admm:
                    if args.fft_epochs != args.num_epochs:
                        if args.distdataparallel:
                            loss_dict[
                                "fft_grad_norm"
                            ] += FFT.module.wiener_crop.grad.norm()
                        else:
                            loss_dict["fft_grad_norm"] += FFT.wiener_crop.grad.norm()
                    else:
                        loss_dict["fft_grad_norm"] += torch.tensor(0.0).to(rank)

                g_optimizer.step()

                # Update lr schedulers
                g_lr_scheduler.step(epoch + i / len(data.train_loader))
                d_lr_scheduler.step(epoch + i / len(data.train_loader))

                if epoch >= args.fft_epochs:
                    fft_optimizer.step()
                    fft_lr_scheduler.step(
                        epoch - args.fft_epochs + i / len(data.train_loader)
                    )

                # Accumulate all losses
                loss_dict["g_loss"] += g_loss.total_loss
                loss_dict["adversarial_loss"] += g_loss.adversarial_loss
                loss_dict["perception_loss"] += g_loss.perception_loss
                loss_dict["contextual_loss"] += g_loss.contextual_loss
                loss_dict["image_loss"] += g_loss.image_loss

                # For test set, reduction is wasteful
                # Much larger, probably evens out
                exp_loss += reduce_loss_dict(loss_dict, world_size=1)
                global_step += args.batch_size * world_size

                if is_local_rank_0:
                    train_pbar.update(args.batch_size)

                    train_pbar.set_description(
                        f"Epoch: {epoch + 1} | Gen loss: {exp_loss.loss_dict['g_loss']:.3f} "
                    )

                    # Write lr rates and metrics
                    if is_local_rank_0 and i % (args.log_interval) == 0:
                        gen_lr = g_optimizer.param_groups[0]["lr"]
                        writer.add_scalar("lr/gen", gen_lr, global_step)

                        disc_lr = d_optimizer.param_groups[0]["lr"]
                        writer.add_scalar("lr/disc", disc_lr, global_step)

                        if epoch >= args.fft_epochs:
                            fft_lr = fft_optimizer.param_groups[0]["lr"]
                            writer.add_scalar(
                                f"lr/{interm_name.lower()}", fft_lr, global_step
                            )

                        for metric in exp_loss.loss_dict:
                            writer.add_scalar(
                                f"Train_Metrics/{metric.replace('fft',interm_name.lower())}",
                                exp_loss.loss_dict[metric],
                                global_step,
                            )

                        n = np.min([3, args.batch_size])
                        for e in range(n):
                            if not is_admm:
                                source_vis = rggb_2_rgb(source[e]).mul(0.5).add(0.5)
                                fft_output_vis = (
                                    rggb_2_rgb(fft_output[e]).mul(0.5).add(0.5)
                                )
                            else:
                                source_vis = source[e].mul(0.5).add(0.5)
                                fft_output_vis = fft_output[e].mul(0.5).add(0.5)

                            fft_output_vis = (fft_output_vis - fft_output_vis.min()) / (
                                fft_output_vis.max() - fft_output_vis.min()
                            )

                            target_vis = target[e].mul(0.5).add(0.5)
                            output_vis = output[e].mul(0.5).add(0.5)

                            writer.add_image(
                                f"Source/Train_{e + 1}",
                                source_vis.cpu().detach(),
                                global_step,
                            )

                            writer.add_image(
                                f"{interm_name}/Train_{e + 1}",
                                fft_output_vis.cpu().detach(),
                                global_step,
                            )

                            writer.add_image(
                                f"Target/Train_{e + 1}",
                                target_vis.cpu().detach(),
                                global_step,
                            )

                            writer.add_image(
                                f"Output/Train_{e + 1}",
                                output_vis.cpu().detach(),
                                global_step,
                            )

                            writer.add_text(
                                f"Filename/Train_{e + 1}", filename[e], global_step
                            )

                # Save checkpoint
                if is_local_rank_0 and (i % args.save_ckpt_interval == 0):

                    logging.info(
                        f"Saving weights at epoch {epoch + 1} global step {global_step}"
                    )

                    # Save weights
                    save_weights(
                        epoch=epoch,
                        global_step=global_step,
                        G=G,
                        D=D,
                        FFT=FFT,
                        g_optimizer=g_optimizer,
                        d_optimizer=d_optimizer,
                        fft_optimizer=fft_optimizer,
                        loss=loss,
                        tag="latest",
                        args=args,
                        is_local_rank_0=True,
                    )

            with torch.no_grad():
                G.eval()
                FFT.eval()
                D.eval()
                filename_static = []
                avg_metrics.reset()

                if is_local_rank_0:
                    val_pbar.reset()

                for i, batch in enumerate(data.val_loader):
                    metrics_dict = defaultdict(float)

                    source, target, filename = batch
                    source, target = (source.to(rank), target.to(rank))

                    n, c, h, w = target.shape
                    target_unpixel_shuffled = unpixel_shuffle(
                        target, args.pixelshuffle_ratio
                    ).reshape(
                        n * args.pixelshuffle_ratio ** 2,
                        c,
                        h // args.pixelshuffle_ratio,
                        w // args.pixelshuffle_ratio,
                    )

                    fft_output = FFT(source)

                    if is_admm:
                        # Upsample
                        fft_output = F.interpolate(
                            fft_output, scale_factor=4, mode="nearest"
                        )

                    # Unpixelshuffle
                    fft_unpixel_shuffled = unpixel_shuffle(
                        fft_output, args.pixelshuffle_ratio
                    )
                    output_unpixel_shuffled = G(fft_unpixel_shuffled)

                    n, c, h, w = output_unpixel_shuffled.shape
                    output_unpixel_shuffled = output_unpixel_shuffled.reshape(
                        n * args.pixelshuffle_ratio ** 2,
                        c // args.pixelshuffle_ratio ** 2,
                        h,
                        w,
                    )

                    fake_logit = D(output_unpixel_shuffled)

                    g_loss(
                        output=output_unpixel_shuffled,
                        target=target_unpixel_shuffled,
                        fake_logit=fake_logit,
                    )

                    n, c, h, w = output_unpixel_shuffled.shape
                    output_unpixel_shuffled = output_unpixel_shuffled.reshape(
                        n // args.pixelshuffle_ratio ** 2,
                        c * args.pixelshuffle_ratio ** 2,
                        h,
                        w,
                    )
                    output = F.pixel_shuffle(
                        output_unpixel_shuffled, args.pixelshuffle_ratio
                    )

                    metrics_dict["g_loss"] += g_loss.total_loss

                    # Save image
                    if args.static_val_image in filename:
                        filename_static = filename
                        source_static = source
                        fft_output_static = fft_output
                        target_static = target
                        output_static = output

                    # PSNR
                    metrics_dict["PSNR"] += PSNR(output, target)
                    avg_metrics += reduce_loss_dict(metrics_dict, world_size=world_size)

                    if is_local_rank_0:
                        val_pbar.update(args.batch_size)
                        val_pbar.set_description(
                            f"Val Epoch : {epoch + 1} Step: {global_step}| PSNR: {avg_metrics.loss_dict['PSNR']:.3f}| Total Loss: {avg_metrics.loss_dict['g_loss']:.3f}"
                        )

                if is_local_rank_0:
                    for metric in avg_metrics.loss_dict:
                        writer.add_scalar(
                            f"Val_Metrics/{metric}",
                            avg_metrics.loss_dict[metric],
                            global_step,
                        )
                    n = np.min([3, args.batch_size])
                    for e in range(n):
                        if not is_admm:
                            source_vis = rggb_2_rgb(source[e]).mul(0.5).add(0.5)
                            fft_output_vis = rggb_2_rgb(fft_output[e]).mul(0.5).add(0.5)
                        else:
                            source_vis = source[e].mul(0.5).add(0.5)
                            fft_output_vis = fft_output[e].mul(0.5).add(0.5)

                        fft_output_vis = (fft_output_vis - fft_output_vis.min()) / (
                            fft_output_vis.max() - fft_output_vis.min()
                        )

                        fft_output_vis = (fft_output_vis - fft_output_vis.min()) / (
                            fft_output_vis.max() - fft_output_vis.min()
                        )

                        target_vis = target[e].mul(0.5).add(0.5)
                        output_vis = output[e].mul(0.5).add(0.5)

                        writer.add_image(
                            f"{interm_name}/Val_{e+1}",
                            fft_output_vis.cpu().detach(),
                            global_step,
                        )

                        writer.add_image(
                            f"Source/Val_{e+1}", source_vis.cpu().detach(), global_step
                        )
                        writer.add_image(
                            f"Target/Val_{e+1}", target_vis.cpu().detach(), global_step
                        )
                        writer.add_image(
                            f"Output/Val_{e+1}", output_vis.cpu().detach(), global_step
                        )

                        writer.add_text(
                            f"Filename/Val_{e + 1}", filename[e], global_step
                        )

                    for e, filename in enumerate(filename_static):
                        if filename == args.static_val_image:
                            if not is_admm:
                                source_vis = (
                                    rggb_2_rgb(source_static[e]).mul(0.5).add(0.5)
                                )
                                fft_output_vis = (
                                    rggb_2_rgb(fft_output_static[e]).mul(0.5).add(0.5)
                                )
                            else:
                                source_vis = source_static[e].mul(0.5).add(0.5)
                                fft_output_vis = fft_output_static[e].mul(0.5).add(0.5)

                            fft_output_vis = (fft_output_vis - fft_output_vis.min()) / (
                                fft_output_vis.max() - fft_output_vis.min()
                            )

                            target_vis = target_static[e].mul(0.5).add(0.5)
                            output_vis = output_static[e].mul(0.5).add(0.5)

                            writer.add_image(
                                f"{interm_name}/Val_Static",
                                fft_output_vis.cpu().detach(),
                                global_step,
                            )

                            writer.add_image(
                                f"Source/Val_Static",
                                source_vis.cpu().detach(),
                                global_step,
                            )
                            writer.add_image(
                                f"Target/Val_Static",
                                target_vis.cpu().detach(),
                                global_step,
                            )
                            writer.add_image(
                                f"Output/Val_Static",
                                output_vis.cpu().detach(),
                                global_step,
                            )

                            writer.add_text(
                                f"Filename/Val_Static", filename[e], global_step
                            )

                            break

                    logging.info(
                        f"Saving weights at END OF epoch {epoch + 1} global step {global_step}"
                    )

                    # Save weights
                    if avg_metrics.loss_dict["g_loss"] < loss:
                        is_min = True
                        loss = avg_metrics.loss_dict["g_loss"]
                    else:
                        is_min = False

                    # Save weights
                    save_weights(
                        epoch=epoch,
                        global_step=global_step,
                        G=G,
                        FFT=FFT,
                        D=D,
                        g_optimizer=g_optimizer,
                        d_optimizer=d_optimizer,
                        fft_optimizer=fft_optimizer,
                        loss=loss,
                        is_min=is_min,
                        args=args,
                        tag="best",
                    )

                # Test
                if not data.test_loader:
                    continue

                filename_static = []
                if is_local_rank_0:
                    test_pbar.reset()

                for i, batch in enumerate(data.test_loader):
                    source, filename = batch
                    source = source.to(rank)

                    fft_output = FFT(source)

                    if is_admm:
                        # Upsample
                        fft_output = F.interpolate(
                            fft_output, scale_factor=4, mode="nearest"
                        )

                    # Unpixelshuffle
                    fft_unpixel_shuffled = unpixel_shuffle(
                        fft_output, args.pixelshuffle_ratio
                    )
                    output_unpixel_shuffled = G(fft_unpixel_shuffled)

                    output = F.pixel_shuffle(
                        output_unpixel_shuffled, args.pixelshuffle_ratio
                    )

                    # Save image
                    if args.static_test_image in filename:
                        filename_static = filename
                        source_static = source
                        fft_output_static = fft_output
                        output_static = output

                    if is_local_rank_0:
                        test_pbar.update(args.batch_size)
                        test_pbar.set_description(
                            f"Test Epoch : {epoch + 1} Step: {global_step}"
                        )

                if is_local_rank_0:
                    n = np.min([3, args.batch_size])
                    for e in range(n):
                        if not is_admm:
                            source_vis = rggb_2_rgb(source[e]).mul(0.5).add(0.5)
                            fft_output_vis = rggb_2_rgb(fft_output[e]).mul(0.5).add(0.5)
                        else:
                            source_vis = source[e].mul(0.5).add(0.5)
                            fft_output_vis = fft_output[e].mul(0.5).add(0.5)

                        fft_output_vis = (fft_output_vis - fft_output_vis.min()) / (
                            fft_output_vis.max() - fft_output_vis.min()
                        )

                        fft_output_vis = (fft_output_vis - fft_output_vis.min()) / (
                            fft_output_vis.max() - fft_output_vis.min()
                        )

                        output_vis = output[e].mul(0.5).add(0.5)

                        writer.add_image(
                            f"{interm_name}/Test_{e + 1}",
                            fft_output_vis.cpu().detach(),
                            global_step,
                        )

                        writer.add_image(
                            f"Source/Test_{e+1}", source_vis.cpu().detach(), global_step
                        )

                        writer.add_image(
                            f"Output/Test_{e+1}", output_vis.cpu().detach(), global_step
                        )

                        writer.add_text(
                            f"Filename/Test_{e + 1}", filename[e], global_step
                        )

                    for e, filename in enumerate(filename_static):
                        if filename == args.static_test_image:
                            if not is_admm:
                                source_vis = (
                                    rggb_2_rgb(source_static[e]).mul(0.5).add(0.5)
                                )
                                fft_output_vis = (
                                    rggb_2_rgb(fft_output_static[e]).mul(0.5).add(0.5)
                                )
                            else:
                                source_vis = source_static[e].mul(0.5).add(0.5)
                                fft_output_vis = fft_output_static[e].mul(0.5).add(0.5)

                            fft_output_vis = (fft_output_vis - fft_output_vis.min()) / (
                                fft_output_vis.max() - fft_output_vis.min()
                            )

                            output_vis = output_static[e].mul(0.5).add(0.5)

                            writer.add_image(
                                f"{interm_name}/Test_Static",
                                fft_output_vis.cpu().detach(),
                                global_step,
                            )

                            writer.add_image(
                                f"Source/Test_Static",
                                source_vis.cpu().detach(),
                                global_step,
                            )

                            writer.add_image(
                                f"Output/Test_Static",
                                output_vis.cpu().detach(),
                                global_step,
                            )

                            writer.add_text(
                                f"Filename/Test_Static", filename[e], global_step
                            )

                            break

                    test_pbar.refresh()

    except KeyboardInterrupt:
        if is_local_rank_0:
            logging.info("-" * 89)
            logging.info("Exiting from training early. Saving models")

            for pbar in [train_pbar, val_pbar, test_pbar]:
                if pbar:
                    pbar.refresh()

            save_weights(
                epoch=epoch,
                global_step=global_step,
                G=G,
                D=D,
                FFT=FFT,
                g_optimizer=g_optimizer,
                d_optimizer=d_optimizer,
                fft_optimizer=fft_optimizer,
                loss=loss,
                is_min=True,
                args=args,
            )
