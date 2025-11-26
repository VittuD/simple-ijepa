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

import hydra
from hydra.utils import get_original_cwd

from train import update_gamma

from simple_ijepa.transformer import VisionTransformer
from simple_ijepa.ijepa import IJEPA
from simple_ijepa.ijepa_gated import GatedIJEPA
from simple_ijepa.stl10_eval import STL10Eval
from simple_ijepa.utils import training_transforms
from simple_ijepa.dataset import MaskedImageDataset, collate_fn

from simple_ijepa.config import TrainConfig


SEED = 42


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
            # "all_open": just use the student tokens directly
            context_tokens = student_tokens

        out = m.predictor(
            context_tokens,
            patchify=False,
            pos_embed=False,
        )
        out = F.layer_norm(out, (out.size(-1),))

        return out


class EncoderAsIJEPA:
    """
    Tiny shim so we can re-use STL10Eval.evaluate(), which expects
    an object with a `.target_encoder` attribute.
    """

    def __init__(self, encoder: torch.nn.Module):
        self.target_encoder = encoder


@hydra.main(version_base=None, config_name="train_config")
def main(cfg: TrainConfig):
    # Make dataset & model dirs relative to the original project root,
    # not Hydra's per-run working directory.
    root = get_original_cwd()
    if not os.path.isabs(cfg.dataset_path):
        cfg.dataset_path = os.path.join(root, cfg.dataset_path)
    if not os.path.isabs(cfg.save_model_dir):
        cfg.save_model_dir = os.path.join(root, cfg.save_model_dir)

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

    if cfg.variant == "baseline":
        model = IJEPA(
            encoder,
            hidden_emb_dim=dim,
            patch_size=patch_size,
            num_targets=num_targets,
        )

        if cfg.ckpt_path is not None:
            if rank == 0:
                print(f"Loading baseline checkpoint from {cfg.ckpt_path}")
            map_location = {"cuda:0": f"cuda:{local_rank}"}
            state = torch.load(cfg.ckpt_path, map_location=map_location)
            model.load_state_dict(state)

        model = model.to(device)

        params = (
            list(model.context_encoder.parameters())
            + [model.mask_token]
            + list(model.predictor.parameters())
        )
        optimizer = torch.optim.Adam(
            params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    else:  # gated
        model = GatedIJEPA(
            encoder=encoder,
            hidden_emb_dim=dim,
            img_size=image_size,
            patch_size=patch_size,
            predictor_depth=depth,
            predictor_heads=heads,
            lambda_gates=cfg.lambda_gates,
            gate_exp_alpha=cfg.gate_exp_alpha,
        )

        if cfg.ckpt_path is not None:
            if rank == 0:
                print(f"Loading gated checkpoint from {cfg.ckpt_path}")
            map_location = {"cuda:0": f"cuda:{local_rank}"}
            state = torch.load(cfg.ckpt_path, map_location=map_location)
            model.load_state_dict(state)

        model = model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    # Wrap with DDP AFTER creating the optimizer
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True,
    )

    # ----------------------------
    # Dataset & DataLoader (DDP)
    # ----------------------------
    if cfg.dataset_name != "stl10":
        raise ValueError("Only STL10 is supported in this implementation.")

    base_ds = STL10(
        cfg.dataset_path,
        split="unlabeled",
        download=(rank == 0),
        transform=training_transforms((image_size, image_size)),
    )

    # Ensure all ranks wait until data is ready
    dist.barrier()

    if cfg.variant == "baseline":
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
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    if cfg.variant == "baseline":
        loader_kwargs["collate_fn"] = collate_fn

    train_loader = DataLoader(**loader_kwargs)

    scaler = GradScaler(enabled=cfg.fp16_precision)

    # Only rank 0 will run linear probing evals & print accuracy
    stl10_eval = (
        STL10Eval(image_size=image_size, dataset_path=cfg.dataset_path)
        if rank == 0
        else None
    )

    total_num_steps = (
        len(train_loader) * (cfg.num_epochs + 2)
    ) - cfg.update_gamma_after_step
    gamma = cfg.gamma
    global_step = 0
    total_loss = 0.0

    if rank == 0:
        variant_str = "baseline I-JEPA" if cfg.variant == "baseline" else "Gated I-JEPA"
        print(
            f"Starting DDP training ({variant_str}) with {world_size} GPUs, "
            f"per-GPU batch size {cfg.batch_size}, "
            f"effective global batch size {cfg.batch_size * world_size}"
        )

    for epoch in range(cfg.num_epochs):
        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0

        if rank == 0:
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{cfg.num_epochs}",
                leave=True,
            )
        else:
            progress_bar = train_loader

        for step, batch in enumerate(progress_bar):
            if cfg.variant == "baseline":
                images, context_indices, target_indices_list = batch
            else:
                images, _ = batch

            images = images.to(device, non_blocking=True)

            with autocast(enabled=cfg.fp16_precision):
                if cfg.variant == "baseline":
                    loss = ddp_model(images, context_indices, target_indices_list)
                    stats = None
                else:
                    loss, stats = ddp_model(images)

            # Gated-only debug visualization on first step of each epoch,
            # controlled by cfg.save_debug_masks
            if (
                cfg.variant == "gated"
                and cfg.save_debug_masks
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
                        save_root=cfg.save_model_dir,
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
                global_step > cfg.update_gamma_after_step
                and global_step % cfg.update_gamma_every_n_steps == 0
            ):
                ddp_model.module.update_params(gamma)
                gamma = update_gamma(global_step, total_num_steps, cfg.gamma)

            if global_step <= cfg.update_gamma_after_step:
                ddp_model.module.copy_params()

            loss_value = loss.item()
            total_loss += loss_value
            epoch_loss += loss_value

            avg_loss = total_loss / (global_step + 1)
            ep_loss = epoch_loss / (step + 1)

            if rank == 0:
                current_lr = optimizer.param_groups[0]["lr"]

                if cfg.variant == "baseline":
                    desc = (
                        f"Epoch {epoch+1}/{cfg.num_epochs} | "
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
                        f"Epoch {epoch+1}/{cfg.num_epochs} | "
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
            if rank == 0 and global_step % cfg.log_every_n_steps == 0:
                os.makedirs(cfg.save_model_dir, exist_ok=True)

                if cfg.variant == "baseline":
                    ckpt_path = os.path.join(cfg.save_model_dir, "training_model_ddp.pth")
                    enc_path = os.path.join(cfg.save_model_dir, "encoder_ddp.pth")
                else:
                    ckpt_path = os.path.join(
                        cfg.save_model_dir, "training_model_gated_ddp.pth"
                    )
                    enc_path = os.path.join(
                        cfg.save_model_dir, "encoder_gated_ddp.pth"
                    )

                torch.save(ddp_model.module.state_dict(), ckpt_path)
                ddp_model.module.save_encoder(enc_path)

            if (
                rank == 0
                and stl10_eval is not None
                and global_step % (cfg.log_every_n_steps * 5) == 0
            ):
                if cfg.variant == "baseline":
                    # Standard path: use teacher encoder as in original code
                    stl10_eval.evaluate(ddp_model.module)
                else:
                    # Reuse STL10Eval by wrapping custom encoders
                    print("Evaluating STL10 with predictor + learned gates...")
                    gated_encoder = GatedPredictorEncoder(
                        ddp_model.module, mode="gated"
                    ).to(stl10_eval.device)
                    stl10_eval.evaluate(EncoderAsIJEPA(gated_encoder))

                    print("Evaluating STL10 with predictor + all gates open...")
                    all_open_encoder = GatedPredictorEncoder(
                        ddp_model.module, mode="all_open"
                    ).to(stl10_eval.device)
                    stl10_eval.evaluate(EncoderAsIJEPA(all_open_encoder))

                print("!" * 100)

    if rank == 0:
        print("Training completed.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
