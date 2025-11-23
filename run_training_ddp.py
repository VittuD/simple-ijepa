# run_training_ddp.py
#
# DDP training script for the *gated* I-JEPA variant (GatedIJEPA).
# Each GPU runs one process, and the model internally decides which
# tokens to keep as context and which to mask via hard-concrete gates.
#
# Observability:
#   The model returns (loss, stats) where `stats` contains:
#     - pred_loss
#     - gate_l0
#     - open_prob_mean
#     - closed_prob_mean
#     - open_frac_per_sample
#     - closed_frac_per_sample
#     - gate_map_example
#
# We log the key scalars in the tqdm description for rank 0.

import argparse
import os
import random
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
from torchvision.datasets import STL10

from train import update_gamma  # reuse the same cosine EMA function

from simple_ijepa.transformer import VisionTransformer
from simple_ijepa.ijepa_gated import GatedIJEPA
from simple_ijepa.stl10_eval import STL10Eval
from simple_ijepa.utils import training_transforms

SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Gated I-JEPA DDP")

    # same interface as run_training.py (mostly retro-compatible)
    parser.add_argument(
        "--dataset_path",
        default="./data",
        help="Path where datasets will be saved",
    )
    parser.add_argument(
        "--dataset_name",
        default="stl10",
        choices=["stl10"],
        help="Dataset name",
    )
    parser.add_argument(
        "-save_model_dir",
        default="./models",
        help="Path where models will be saved",
    )
    parser.add_argument(
        "--num_epochs",
        default=100,
        type=int,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=256,
        type=int,
        help="Per-process batch size (per GPU)",
    )
    parser.add_argument(
        "-lr", "--learning_rate", default=3e-4, type=float
    )
    parser.add_argument(
        "-wd", "--weight_decay", default=1e-5, type=float
    )
    parser.add_argument(
        "--fp16_precision",
        action="store_true",
        help="Use 16-bit precision for GPU training",
    )
    parser.add_argument(
        "--emb_dim",
        default=768,
        type=int,
        help="Transformer embedding dimm (unused here, fixed 512 in code)",
    )
    parser.add_argument(
        "--log_every_n_steps",
        default=200,
        type=int,
        help="Log / save every n steps (global steps, not epochs)",
    )
    parser.add_argument(
        "--gamma",
        default=0.996,
        type=float,
        help="Initial EMA coefficient",
    )
    parser.add_argument(
        "--update_gamma_after_step",
        default=1,
        type=int,
        help="Update EMA gamma after this step",
    )
    parser.add_argument(
        "--update_gamma_every_n_steps",
        default=1,
        type=int,
        help="Update EMA gamma after this many steps",
    )
    parser.add_argument(
        "--ckpt_path",
        default=None,
        type=str,
        help="Path to training_model.pth to resume training",
    )

    # optional extras, backward compatible (only for DDP script)
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of dataloader workers per process",
    )

    # gating-specific hyperparameter (you can tune this)
    parser.add_argument(
        "--lambda_gates",
        default=1e-3,
        type=float,
        help="Weight for the gate L0 penalty term.",
    )

    return parser.parse_args()


def setup_distributed():
    """
    Initialize the default process group using torchrun's env variables.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def set_seed(seed, rank):
    full_seed = seed + rank
    random.seed(full_seed)
    np.random.seed(full_seed)
    torch.manual_seed(full_seed)
    torch.cuda.manual_seed_all(full_seed)


def main():
    args = parse_args()

    rank, local_rank, world_size = setup_distributed()
    set_seed(SEED, rank)

    device = torch.device("cuda", local_rank)

    # ----------------------------
    # Model setup (similar to train.py)
    # ----------------------------
    dim = 512
    image_size = 96
    patch_size = 8
    depth = 6
    heads = 6
    mlp_dim = dim * 2

    encoder = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
    )

    model = GatedIJEPA(
        encoder=encoder,
        hidden_emb_dim=dim,
        img_size=image_size,
        patch_size=patch_size,
        predictor_depth=6,
        predictor_heads=6,
        lambda_gates=args.lambda_gates,
    )

    if args.ckpt_path is not None:
        if rank == 0:
            print(f"Loading checkpoint from {args.ckpt_path}")
        map_location = {"cuda:0": f"cuda:{local_rank}"}
        model_state = torch.load(args.ckpt_path, map_location=map_location)
        model.load_state_dict(model_state)

    model = model.to(device)

    # Optimizer over context encoder + mask token + predictor + gates
    params = list(model.parameters())
    optimizer = torch.optim.Adam(
        params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Wrap with DDP AFTER creating the optimizer
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    # ----------------------------
    # Dataset & DataLoader (DDP)
    # ----------------------------
    if args.dataset_name != "stl10":
        raise ValueError("Only STL10 is supported in this implementation.")

    stl10_ds = STL10(
        args.dataset_path,
        split="unlabeled",
        download=(rank == 0),  # only rank0 downloads; others will reuse files
        transform=training_transforms((image_size, image_size)),
    )

    # make sure all ranks wait until data is ready
    dist.barrier()

    train_sampler = DistributedSampler(
        stl10_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    )

    train_loader = DataLoader(
        stl10_ds,
        batch_size=args.batch_size,  # per-GPU batch size
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    scaler = GradScaler(enabled=args.fp16_precision)

    # Only rank 0 will run linear probing evals & print accuracy
    stl10_eval = STL10Eval() if rank == 0 else None

    total_num_steps = (
        len(train_loader) * (args.num_epochs + 2)
    ) - args.update_gamma_after_step
    gamma = args.gamma
    global_step = 0
    total_loss = 0.0

    if rank == 0:
        print(
            f"Starting Gated I-JEPA DDP training with {world_size} GPUs, "
            f"per-GPU batch size {args.batch_size}, "
            f"effective global batch size {args.batch_size * world_size}"
        )

    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)

        epoch_loss = 0.0

        if rank == 0:
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{args.num_epochs}",
                leave=True,
            )
        else:
            progress_bar = train_loader

        for step, (images, _) in enumerate(progress_bar):
            images = images.to(device, non_blocking=True)

            with autocast(enabled=args.fp16_precision):
                loss, stats = ddp_model(images)
                # stats is a dict with keys:
                #   pred_loss, gate_l0, open_prob_mean, closed_prob_mean, ...

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # EMA update schedule (same logic as train.py), but only for encoders
            if (
                global_step > args.update_gamma_after_step
                and global_step % args.update_gamma_every_n_steps == 0
            ):
                ddp_model.module.update_params(gamma)
                gamma = update_gamma(global_step, total_num_steps, args.gamma)

            if global_step <= args.update_gamma_after_step:
                ddp_model.module.copy_params()

            # For logging, we can use per-rank loss; rank 0 prints.
            loss_value = loss.item()
            total_loss += loss_value
            epoch_loss += loss_value

            avg_loss = total_loss / (global_step + 1)
            ep_loss = epoch_loss / (step + 1)

            if rank == 0:
                current_lr = optimizer.param_groups[0]["lr"]

                # Pull some scalars from stats
                pred_loss = stats["pred_loss"].item()
                gate_l0 = stats["gate_l0"].item()
                open_prob_mean = stats["open_prob_mean"].item()
                closed_prob_mean = stats["closed_prob_mean"].item()

                progress_bar.set_description(
                    f"Epoch {epoch+1}/{args.num_epochs} | "
                    f"Step {global_step+1} | "
                    f"EpLoss: {ep_loss:.4f} | "
                    f"TotLoss: {avg_loss:.4f} | "
                    f"Pred: {pred_loss:.4f} | "
                    f"L0: {gate_l0:.2f} | "
                    f"Open: {open_prob_mean*100:.1f}% | "
                    f"Closed: {closed_prob_mean*100:.1f}% | "
                    f"EMA γ: {gamma:.4f} | "
                    f"Lr: {current_lr:.6f}"
                )

            global_step += 1

            # Checkpointing & eval only on rank 0
            if rank == 0 and global_step % args.log_every_n_steps == 0:
                os.makedirs(args.save_model_dir, exist_ok=True)
                ckpt_path = os.path.join(args.save_model_dir, "training_model_gated_ddp.pth")
                enc_path = os.path.join(args.save_model_dir, "encoder_gated_ddp.pth")

                torch.save(ddp_model.module.state_dict(), ckpt_path)
                ddp_model.module.save_encoder(enc_path)

            if (
                rank == 0
                and stl10_eval is not None
                and global_step % (args.log_every_n_steps * 5) == 0
            ):
                # Evaluate target encoder with linear probe
                stl10_eval.evaluate(ddp_model.module)
                print("!" * 100)

    if rank == 0:
        print("Training completed.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
