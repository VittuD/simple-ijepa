import argparse
import os
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
from torchvision.datasets import STL10
from torchvision.utils import save_image

from train import update_gamma

from simple_ijepa.transformer import VisionTransformer
from simple_ijepa.ijepa import IJEPA
from simple_ijepa.ijepa_gated import GatedIJEPA
from simple_ijepa.stl10_eval import STL10Eval, logistic_regression
from simple_ijepa.utils import training_transforms
from simple_ijepa.dataset import MaskedImageDataset, collate_fn


SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="I-JEPA DDP (baseline or gated variant)"
    )

    parser.add_argument(
        "--variant",
        default="baseline",
        choices=["baseline", "gated"],
        help="Which model variant to train: 'baseline' (IJEPA) or 'gated' (GatedIJEPA).",
    )

    # Common args (kept backward compatible with existing scripts)
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
        "-lr",
        "--learning_rate",
        default=3e-4,
        type=float,
    )
    parser.add_argument(
        "-wd",
        "--weight_decay",
        default=1e-5,
        type=float,
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

    # Dataloader workers (both variants)
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of dataloader workers per process",
    )

    # Gated-only knobs (ignored for baseline)
    parser.add_argument(
        "--lambda_gates",
        default=1e-3,
        type=float,
        help="[gated] Weight for the gate penalty term.",
    )
    parser.add_argument(
        "--gate_exp_alpha",
        default=5.0,
        type=float,
        help="[gated] Sharpness of exponential gate penalty as open fraction -> 1.",
    )

    return parser.parse_args()


def setup_distributed():
    """Initialize the default process group using torchrun's env variables."""
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


# -----------------------------
# Gated-specific utilities
# -----------------------------
def save_debug_masks(
    images: torch.Tensor,
    gate_values_full: torch.Tensor,
    epoch: int,
    global_step: int,
    save_root: str,
    image_size: int,
    patch_size: int,
    max_images: int = 8,
):
    """
    Save original and patch-masked versions of the first few images in the batch,
    using the gate values to decide which patches to "mask".
    """
    os.makedirs(os.path.join(save_root, "debug_masks"), exist_ok=True)
    debug_dir = os.path.join(save_root, "debug_masks")

    B, C, H, W = images.shape
    assert H == image_size and W == image_size, "Unexpected image size."

    num_to_save = min(max_images, B)
    H_p = image_size // patch_size
    W_p = image_size // patch_size
    N = H_p * W_p

    assert gate_values_full.shape[0] == B
    assert gate_values_full.shape[1] == N, "gate_values_full has wrong length."

    images_cpu = images[:num_to_save].detach().cpu()
    gates_cpu = gate_values_full[:num_to_save].detach().cpu()  # (K, N)

    masked_images = []

    for idx in range(num_to_save):
        img = images_cpu[idx].clone()
        g = gates_cpu[idx]

        gate_map = g.view(H_p, W_p)

        img_masked = img.clone()

        for ph in range(H_p):
            for pw in range(W_p):
                if gate_map[ph, pw] < 0.5:
                    h0 = ph * patch_size
                    h1 = h0 + patch_size
                    w0 = pw * patch_size
                    w1 = w0 + patch_size

                    img_masked[:, h0:h1, w0:w1] = 0.0

                    size = patch_size
                    diag = torch.arange(size)

                    img_masked[:, h0 + diag, w0 + diag] = 0.7
                    img_masked[:, h0 + diag, w1 - 1 - diag] = 0.7

        masked_images.append(img_masked)

    masked_batch = torch.stack(masked_images, dim=0)

    step_str = f"e{epoch+1:03d}_s{global_step+1:06d}"
    orig_path = os.path.join(debug_dir, f"{step_str}_orig.png")
    masked_path = os.path.join(debug_dir, f"{step_str}_masked.png")

    save_image(
        images_cpu,
        orig_path,
        nrow=num_to_save,
        normalize=True,
        value_range=(0.0, 1.0),
    )
    save_image(
        masked_batch,
        masked_path,
        nrow=num_to_save,
        normalize=True,
        value_range=(0.0, 1.0),
    )


class GatedPredictorEncoder(torch.nn.Module):
    """
    Wraps a GatedIJEPA model to expose a plain encoder interface:
        forward(x) -> (B, N, D) token embeddings

    mode = "gated":
        Uses the learned gates to mix student tokens and mask token,
        then runs the predictor.

    mode = "all_open":
        Uses the raw student tokens (no mask tokens), then runs the predictor.
    """

    def __init__(self, ijepa_model, mode: str = "gated"):
        super().__init__()
        assert mode in ("gated", "all_open")
        self.ijepa = ijepa_model
        self.mode = mode

    @torch.inference_mode
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        m = self.ijepa

        student_tokens = m.context_encoder(x)  # (B, N, D)
        B, N, D = student_tokens.shape

        if self.mode == "gated":
            log_alpha = m.gate_mlp(student_tokens).squeeze(-1)  # (B, N)
            gate_values, _ = m.gate(log_alpha, training=False)  # (B, N)
            r = gate_values.unsqueeze(-1)

            mask_token = m.mask_token.to(device)
            mask_tokens = mask_token.expand(B, N, D)
            one_minus_r = 1.0 - r

            context_tokens = r * student_tokens + one_minus_r * mask_tokens
        else:
            context_tokens = student_tokens

        out = m.predictor(
            context_tokens,
            patchify=False,
            pos_embed=False,
        )
        out = F.layer_norm(out, (out.size(-1),))

        return out


@torch.inference_mode
def get_image_embs_labels(encoder, dataloader, device):
    embs, labels = [], []

    for images, targets in dataloader:
        images = images.to(device)
        out = encoder(images)  # (B, N, D)
        features = out.mean(dim=1)

        embs.extend(features.cpu().tolist())
        labels.extend(targets.cpu().tolist())

    return np.array(embs), np.array(labels)


@torch.inference_mode
def evaluate_gated_and_all_open(stl10_eval, ijepa_model):
    device = stl10_eval.device
    train_loader = stl10_eval.train_loader
    val_loader = stl10_eval.val_loader

    gated_encoder = GatedPredictorEncoder(ijepa_model, mode="gated").to(device)
    print("Evaluating STL10 with predictor + learned gates...")
    emb_tr, lab_tr = get_image_embs_labels(gated_encoder, train_loader, device)
    emb_val, lab_val = get_image_embs_labels(gated_encoder, val_loader, device)
    logistic_regression(emb_tr, lab_tr, emb_val, lab_val)

    all_open_encoder = GatedPredictorEncoder(ijepa_model, mode="all_open").to(device)
    print("Evaluating STL10 with predictor + all gates open...")
    emb_tr2, lab_tr2 = get_image_embs_labels(all_open_encoder, train_loader, device)
    emb_val2, lab_val2 = get_image_embs_labels(all_open_encoder, val_loader, device)
    logistic_regression(emb_tr2, lab_tr2, emb_val2, lab_val2)


def main():
    args = parse_args()

    rank, local_rank, world_size = setup_distributed()
    set_seed(SEED, rank)

    device = torch.device("cuda", local_rank)

    # Shared model hyper-params
    dim = 512
    image_size = 96
    patch_size = 8
    depth = 6
    heads = 6
    mlp_dim = dim * 2
    num_targets = 4

    # ----------------------------
    # Model setup
    # ----------------------------
    encoder = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
    )

    if args.variant == "baseline":
        model = IJEPA(
            encoder,
            hidden_emb_dim=dim,
            patch_size=patch_size,
            num_targets=num_targets,
        )

        if args.ckpt_path is not None:
            if rank == 0:
                print(f"Loading baseline checkpoint from {args.ckpt_path}")
            map_location = {"cuda:0": f"cuda:{local_rank}"}
            state = torch.load(args.ckpt_path, map_location=map_location)
            model.load_state_dict(state)

        model = model.to(device)

        params = (
            list(model.context_encoder.parameters())
            + [model.mask_token]
            + list(model.predictor.parameters())
        )
        optimizer = torch.optim.Adam(
            params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    else:  # gated
        model = GatedIJEPA(
            encoder=encoder,
            hidden_emb_dim=dim,
            img_size=image_size,
            patch_size=patch_size,
            predictor_depth=depth,
            predictor_heads=heads,
            lambda_gates=args.lambda_gates,
            gate_exp_alpha=args.gate_exp_alpha,
        )

        if args.ckpt_path is not None:
            if rank == 0:
                print(f"Loading gated checkpoint from {args.ckpt_path}")
            map_location = {"cuda:0": f"cuda:{local_rank}"}
            state = torch.load(args.ckpt_path, map_location=map_location)
            model.load_state_dict(state)

        model = model.to(device)

        # Optimizer over all trainable params (teacher encoder has requires_grad=False)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    # Wrap with DDP AFTER creating the optimizer
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,      # no need to track unused params
        gradient_as_bucket_view=True,      # reuse bucket buffers as grads
        static_graph=True,                 # graph structure doesn't change across steps
    )

    # ----------------------------
    # Dataset & DataLoader (DDP)
    # ----------------------------
    if args.dataset_name != "stl10":
        raise ValueError("Only STL10 is supported in this implementation.")

    base_ds = STL10(
        args.dataset_path,
        split="unlabeled",
        download=(rank == 0),
        transform=training_transforms((image_size, image_size)),
    )

    # Ensure all ranks wait until data is ready
    dist.barrier()

    if args.variant == "baseline":
        num_patches = int((image_size // patch_size)) ** 2
        dataset = MaskedImageDataset(
            base_ds,
            num_patches=num_patches,
            num_targets=num_targets,
        )
    else:
        dataset = base_ds

    train_sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    )

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.variant == "baseline":
        loader_kwargs["collate_fn"] = collate_fn

    train_loader = DataLoader(**loader_kwargs)

    scaler = GradScaler(enabled=args.fp16_precision)

    # Only rank 0 will run linear probing evals & print accuracy
    stl10_eval = STL10Eval(image_size=image_size,
                       dataset_path=args.dataset_path) if rank == 0 else None

    total_num_steps = (
        len(train_loader) * (args.num_epochs + 2)
    ) - args.update_gamma_after_step
    gamma = args.gamma
    global_step = 0
    total_loss = 0.0

    if rank == 0:
        variant_str = "baseline I-JEPA" if args.variant == "baseline" else "Gated I-JEPA"
        print(
            f"Starting DDP training ({variant_str}) with {world_size} GPUs, "
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

        for step, batch in enumerate(progress_bar):
            if args.variant == "baseline":
                images, context_indices, target_indices_list = batch
            else:
                images, _ = batch

            images = images.to(device, non_blocking=True)

            with autocast(enabled=args.fp16_precision):
                if args.variant == "baseline":
                    loss = ddp_model(images, context_indices, target_indices_list)
                    stats = None
                else:
                    loss, stats = ddp_model(images)

            # Gated-only debug visualization on first step of each epoch
            if (
                args.variant == "gated"
                and rank == 0
                and step == 0
                and "gate_values_full" in (stats or {})
            ):
                try:
                    save_debug_masks(
                        images=images,
                        gate_values_full=stats["gate_values_full"],
                        epoch=epoch,
                        global_step=global_step,
                        save_root=args.save_model_dir,
                        image_size=image_size,
                        patch_size=patch_size,
                        max_images=8,
                    )
                except Exception as e:
                    print(f"[WARN] Failed to save debug masks: {e}")

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (
                global_step > args.update_gamma_after_step
                and global_step % args.update_gamma_every_n_steps == 0
            ):
                ddp_model.module.update_params(gamma)
                gamma = update_gamma(global_step, total_num_steps, args.gamma)

            if global_step <= args.update_gamma_after_step:
                ddp_model.module.copy_params()

            loss_value = loss.item()
            total_loss += loss_value
            epoch_loss += loss_value

            avg_loss = total_loss / (global_step + 1)
            ep_loss = epoch_loss / (step + 1)

            if rank == 0:
                current_lr = optimizer.param_groups[0]["lr"]

                if args.variant == "baseline":
                    desc = (
                        f"Epoch {epoch+1}/{args.num_epochs} | "
                        f"Step {global_step+1} | "
                        f"Epoch Loss: {ep_loss:.7f} | "
                        f"Total Loss: {avg_loss:.7f} | "
                        f"EMA gamma: {gamma:.6f} | "
                        f"Lr: {current_lr:.6f}"
                    )
                else:
                    pred_loss = stats["pred_loss"].item()
                    gate_penalty = stats["gate_penalty"].item()
                    open_prob_mean = stats["open_prob_mean"].item()
                    closed_prob_mean = stats["closed_prob_mean"].item()

                    desc = (
                        f"Epoch {epoch+1}/{args.num_epochs} | "
                        f"Step {global_step+1} | "
                        f"EpLoss: {ep_loss:.4f} | "
                        f"TotLoss: {avg_loss:.4f} | "
                        f"Pred: {pred_loss:.4f} | "
                        f"GatePen: {gate_penalty:.3f} | "
                        f"Open: {open_prob_mean*100:.1f}% | "
                        f"Closed: {closed_prob_mean*100:.1f}% | "
                        f"EMA γ: {gamma:.4f} | "
                        f"Lr: {current_lr:.6f}"
                    )

                progress_bar.set_description(desc)

            global_step += 1

            # Checkpointing & eval only on rank 0
            if rank == 0 and global_step % args.log_every_n_steps == 0:
                os.makedirs(args.save_model_dir, exist_ok=True)

                if args.variant == "baseline":
                    ckpt_path = os.path.join(args.save_model_dir, "training_model_ddp.pth")
                    enc_path = os.path.join(args.save_model_dir, "encoder_ddp.pth")
                else:
                    ckpt_path = os.path.join(
                        args.save_model_dir, "training_model_gated_ddp.pth"
                    )
                    enc_path = os.path.join(
                        args.save_model_dir, "encoder_gated_ddp.pth"
                    )

                torch.save(ddp_model.module.state_dict(), ckpt_path)
                ddp_model.module.save_encoder(enc_path)

            if (
                rank == 0
                and stl10_eval is not None
                and global_step % (args.log_every_n_steps * 5) == 0
            ):
                if args.variant == "baseline":
                    stl10_eval.evaluate(ddp_model.module)
                else:
                    evaluate_gated_and_all_open(stl10_eval, ddp_model.module)

                print("!" * 100)

    if rank == 0:
        print("Training completed.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
