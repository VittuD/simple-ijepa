# simple_ijepa/simple_ijepa/training/ddp_trainer.py

import os
from typing import Tuple, Dict, Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
from torchvision.datasets import STL10
from torch.optim.lr_scheduler import LambdaLR

from simple_ijepa.transformer import VisionTransformer
from simple_ijepa.ijepa import IJEPA
from simple_ijepa.ijepa_gated import GatedIJEPA
from simple_ijepa.stl10_eval import STL10Eval
from simple_ijepa.utils import (
    training_transforms,
    compute_global_grad_norm,
    compute_global_param_norm,
    summarize_open_fraction,
    summarize_gate_values,
)
from simple_ijepa.dataset import MaskedImageDataset, collate_fn

from simple_ijepa.config import TrainConfig
from simple_ijepa.configuration_ijepa import IJEPAConfig
from simple_ijepa.training.wandb_utils import (
    init_wandb,
    log_metrics,
    log_debug_artifacts,
)
from simple_ijepa.training.dist_utils import (
    maybe_run_eval,
    prepare_paths,
    save_checkpoint,
    set_seed,
    setup_distributed,
    setup_logging,
)
from simple_ijepa.training.schedulers import get_scheduler
from simple_ijepa.model_card import build_model_card


SEED = 42


class IJEPATrainerDDP:
    def __init__(self, cfg: TrainConfig):
        self.cfg = prepare_paths(cfg)

    # ------------------------------------------------------------------
    # Model / optimizer / data builders
    # ------------------------------------------------------------------
    def _build_model_and_optimizer(
        self,
        device: torch.device,
        local_rank: int,
        rank: int,
        model_cfg: IJEPAConfig,
        logger,
    ) -> Tuple[torch.nn.Module, torch.optim.Optimizer, str, str]:
        """Build the encoder + IJEPA or GatedIJEPA model, load an optional
        checkpoint, and create the optimizer.

        Returns:
            model (on correct device),
            optimizer,
            training_checkpoint_filename,
            encoder_checkpoint_filename
        """
        cfg = self.cfg
        encoder = VisionTransformer(
            image_size=model_cfg.image_size,
            patch_size=model_cfg.patch_size,
            dim=model_cfg.hidden_dim,
            depth=model_cfg.depth,
            heads=model_cfg.heads,
            mlp_dim=model_cfg.hidden_dim * 2,
        )

        map_location = {"cuda:0": f"cuda:{local_rank}"}

        if cfg.variant == "baseline":
            model = IJEPA(
                encoder,
                hidden_emb_dim=model_cfg.hidden_dim,
                img_size=model_cfg.image_size,
                patch_size=model_cfg.patch_size,
                num_targets=model_cfg.num_targets,
                predictor_depth=model_cfg.predictor_depth,
                predictor_heads=model_cfg.predictor_heads,
            )

            if cfg.logging.ckpt_path is not None and rank == 0:
                msg = f"Loading baseline checkpoint from {cfg.logging.ckpt_path}"
                logger.info(msg)
                state = torch.load(cfg.logging.ckpt_path, map_location=map_location)
                model.load_state_dict(state)

            model = model.to(device)

            params = (
                list(model.context_encoder.parameters())
                + [model.mask_token]
                + list(model.predictor.parameters())
            )
            optimizer = torch.optim.Adam(
                params,
                lr=cfg.optim.learning_rate,
                weight_decay=cfg.optim.weight_decay,
            )

            training_ckpt_name = "training_model_ddp.pth"
            encoder_ckpt_name = "encoder_ddp.pth"

        else:  # gated
            model = GatedIJEPA(
                encoder=encoder,
                hidden_emb_dim=model_cfg.hidden_dim,
                img_size=model_cfg.image_size,
                patch_size=model_cfg.patch_size,
                predictor_depth=model_cfg.predictor_depth,
                predictor_heads=model_cfg.predictor_heads,
                lambda_gates=model_cfg.lambda_gates,
                gate_beta=model_cfg.gate_beta,
                gate_gamma=model_cfg.gate_gamma,
                gate_zeta=model_cfg.gate_zeta,
                gate_exp_alpha=model_cfg.gate_exp_alpha,
                gate_layer_index=model_cfg.gate_layer_index,
                gate_location=model_cfg.gate_location,
            )

            if cfg.logging.ckpt_path is not None and rank == 0:
                msg = f"Loading gated checkpoint from {cfg.logging.ckpt_path}"
                logger.info(msg)
                state = torch.load(cfg.logging.ckpt_path, map_location=map_location)
                model.load_state_dict(state)

            model = model.to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.optim.learning_rate,
                weight_decay=cfg.optim.weight_decay,
            )

            training_ckpt_name = "training_model_gated_ddp.pth"
            encoder_ckpt_name = "encoder_gated_ddp.pth"

        return model, optimizer, training_ckpt_name, encoder_ckpt_name

    def _build_dataloader(
        self,
        model_cfg: IJEPAConfig,
        world_size: int,
        rank: int,
    ) -> Tuple[DataLoader, DistributedSampler]:
        """Create STL10-based training dataset and DDP-aware DataLoader."""
        cfg = self.cfg
        if cfg.data.dataset_name != "stl10":
            raise ValueError("Only STL10 is supported in this implementation.")

        base_ds = STL10(
            cfg.data.dataset_path,
            split="unlabeled",
            download=(rank == 0),
            transform=training_transforms(
                (model_cfg.image_size, model_cfg.image_size)
            ),
        )

        # Ensure all ranks wait until data is ready
        dist.barrier()

        if cfg.variant == "baseline":
            num_patches = int((model_cfg.image_size // model_cfg.patch_size)) ** 2
            dataset = MaskedImageDataset(
                base_ds,
                num_patches=num_patches,
                num_targets=model_cfg.num_targets,  # must match num_targets passed to IJEPA
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
            batch_size=cfg.dataloader.batch_size,
            sampler=train_sampler,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=True,
            persistent_workers=(cfg.dataloader.num_workers > 0),
            prefetch_factor=4,
        )
        if cfg.variant == "baseline":
            loader_kwargs["collate_fn"] = collate_fn

        train_loader = DataLoader(**loader_kwargs)
        return train_loader, train_sampler

    # ------------------------------------------------------------------
    # Forward wrapper
    # ------------------------------------------------------------------
    def _forward_batch(
        self,
        ddp_model: DDP,
        batch,
        device: torch.device,
        fp16: bool,
    ):
        """Run a forward pass for a batch and return (loss, stats, images)."""
        cfg = self.cfg
        if cfg.variant == "baseline":
            images, context_indices, target_indices_list = batch
        else:
            images, _ = batch

        images = images.to(device, non_blocking=True)

        with autocast(enabled=fp16):
            if cfg.variant == "baseline":
                loss = ddp_model(images, context_indices, target_indices_list)
                stats = None
            else:
                loss, stats = ddp_model(images)

        return loss, stats, images

    # ------------------------------------------------------------------
    # Logging helpers (per-step)
    # ------------------------------------------------------------------
    def _build_gating_metrics(
        self,
        stats: Dict[str, Any],
        model_cfg: IJEPAConfig,
    ) -> Dict[str, float]:
        """Build the gating-related metrics dict from the stats returned
        by GatedIJEPA."""
        pred_loss = stats["pred_loss"].item()
        gate_penalty = stats["gate_penalty"].item()
        open_prob_mean = stats["open_prob_mean"].item()
        closed_prob_mean = stats["closed_prob_mean"].item()

        metrics: Dict[str, float] = {
            "gates/pred_loss": pred_loss,
            "gates/gate_penalty": gate_penalty,
            "gates/open_prob_mean": open_prob_mean,
            "gates/closed_prob_mean": closed_prob_mean,
        }

        # Extra summaries over per-sample fractions and per-gate values
        open_frac_stats = {}
        gate_value_stats = {}

        open_frac_tensor = stats.get("open_frac_per_sample")
        gate_values_full = stats.get("gate_values_full")

        if isinstance(open_frac_tensor, torch.Tensor):
            open_frac_stats = summarize_open_fraction(open_frac_tensor)

        if isinstance(gate_values_full, torch.Tensor):
            gate_value_stats = summarize_gate_values(gate_values_full)

        if open_frac_stats:
            metrics.update(
                {
                    "gates/open_frac_mean": open_frac_stats["mean"],
                    "gates/open_frac_std": open_frac_stats["std"],
                    "gates/open_frac_min": open_frac_stats["min"],
                    "gates/open_frac_max": open_frac_stats["max"],
                }
            )

        if gate_value_stats:
            metrics.update(
                {
                    "gates/gate_value_mean": gate_value_stats["mean"],
                    "gates/gate_value_std": gate_value_stats["std"],
                }
            )

        return metrics

    def _build_step_desc_and_metrics(
        self,
        *,
        epoch: int,
        num_epochs: int,
        global_step: int,
        loss_value: float,
        epoch_loss_avg: float,
        total_loss_avg: float,
        gamma: float,
        optimizer: torch.optim.Optimizer,
        grad_norm: float,
        param_norm: float,
        stats: Dict[str, Any] | None,
        model_cfg: IJEPAConfig,
    ) -> Tuple[str, Dict[str, float]]:
        """Build the progress-bar description and metrics dict for the step."""
        cfg = self.cfg
        current_lr = optimizer.param_groups[0]["lr"]
        current_weight_decay = optimizer.param_groups[0].get("weight_decay", 0.0)

        base_metrics: Dict[str, float] = {
            # Training progress
            "train/epoch": float(epoch + 1),
            "train/global_step": float(global_step + 1),
            "train/loss_step": float(loss_value),
            "train/loss_epoch": float(epoch_loss_avg),
            "train/loss_avg": float(total_loss_avg),
            # Optimization-related
            "optim/ema_gamma": float(gamma),
            "optim/lr": float(current_lr),
            "optim/grad_norm_global": float(grad_norm),
            "optim/param_norm_global": float(param_norm),
            "optim/weight_decay": float(current_weight_decay),
        }

        # Baseline variant: no gating
        if cfg.variant == "baseline":
            desc = (
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Step {global_step+1} | "
                f"Epoch Loss: {epoch_loss_avg:.7f} | "
                f"Total Loss: {total_loss_avg:.7f} | "
                f"EMA γ: {gamma:.6f} | "
                f"Lr: {current_lr:.6f} | "
                f"gN: {grad_norm:.2e} | "
                f"pN: {param_norm:.2e}"
            )
            return desc, base_metrics

        # Gated variant: add gate-specific metrics
        assert stats is not None, "Gated variant expects non-None stats."

        gating_metrics = self._build_gating_metrics(stats, model_cfg=model_cfg)
        metrics = {**base_metrics, **gating_metrics}

        pred_loss = stats["pred_loss"].item()
        gate_penalty = stats["gate_penalty"].item()
        open_prob_mean = stats["open_prob_mean"].item()
        closed_prob_mean = stats["closed_prob_mean"].item()

        desc = (
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Step {global_step+1} | "
            f"EpLoss: {epoch_loss_avg:.4f} | "
            f"TotLoss: {total_loss_avg:.4f} | "
            f"Pred: {pred_loss:.4f} | "
            f"GatePen: {gate_penalty:.3f} | "
            f"Open: {open_prob_mean*100:.1f}% | "
            f"Closed: {closed_prob_mean*100:.1f}% | "
            f"EMA γ: {gamma:.4f} | "
            f"Lr: {current_lr:.6f} | "
            f"gN: {grad_norm:.2e} | "
            f"pN: {param_norm:.2e}"
        )
        return desc, metrics

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(self) -> None:
        cfg = self.cfg

        rank, local_rank, world_size = setup_distributed()
        set_seed(SEED, rank)

        # Process-aware logger (console + file)
        logger = setup_logging(self.cfg, rank)

        # Initialize W&B (only on rank 0; returns None otherwise)
        wandb_run = init_wandb(cfg, rank=rank, world_size=world_size, logger=logger)

        device = torch.device("cuda", local_rank)
        model_cfg = self.cfg.model

        # Model & optimizer
        model, optimizer, training_ckpt_name, encoder_ckpt_name = self._build_model_and_optimizer(
            device,
            local_rank,
            rank,
            model_cfg,
            logger=logger,
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

        # Dataset & loader
        train_loader, train_sampler = self._build_dataloader(
            model_cfg=model_cfg,
            world_size=world_size,
            rank=rank,
        )

        scaler = GradScaler(enabled=cfg.optim.fp16_precision)

        # Only rank 0 will run linear probing evals & log accuracy
        stl10_eval = (
            STL10Eval(
                image_size=model_cfg.image_size,
                dataset_path=cfg.data.dataset_path,
                logger=logger,
                wandb_run=wandb_run,
            )
            if rank == 0
            else None
        )

        # --- EMA scheduler (unchanged, but uses factory) ---
        total_num_steps = (len(train_loader) * (cfg.optim.num_epochs + 2)) - cfg.ema.update_gamma_after_step
        ema_scheduler = get_scheduler(
            name="ema_cosine",
            num_training_steps=total_num_steps,
            ema_base_gamma=cfg.ema.gamma,
        )

        # --- LR + WD scheduler (Hydra-configurable) ---
        num_training_steps_lr = len(train_loader) * cfg.optim.num_epochs
        lr_sched_cfg = cfg.optim.lr_scheduler

        lr_scheduler: LambdaLR = get_scheduler(
            name=lr_sched_cfg.name,
            optimizer=optimizer,
            num_warmup_steps=lr_sched_cfg.num_warmup_steps,
            num_training_steps=num_training_steps_lr,
            num_steps=(lr_sched_cfg.num_steps if lr_sched_cfg.name == "step" else None),
            gamma=(lr_sched_cfg.gamma if lr_sched_cfg.name == "step" else None),
            num_cycles=lr_sched_cfg.num_cycles,
        )  # type: ignore[assignment]

        # Base LR/WD used to derive schedules
        # (we assume a single scalar LR/WD across param groups for now)
        base_lr = optimizer.param_groups[0]["lr"]
        base_weight_decay = optimizer.param_groups[0].get("weight_decay", 0.0)

        gamma = cfg.ema.gamma
        global_step = 0
        total_loss = 0.0

        if rank == 0:
            variant_str = "baseline I-JEPA" if cfg.variant == "baseline" else "Gated I-JEPA"
            logger.info(
                "Starting DDP training (%s) with %d GPUs, per-GPU batch size %d, "
                "effective global batch size %d",
                variant_str,
                world_size,
                cfg.dataloader.batch_size,
                cfg.dataloader.batch_size * world_size,
            )
            if cfg.variant == "gated":
                logger.info(
                    "  lambda_gates = %s, gate_exp_alpha = %s",
                    model_cfg.lambda_gates,
                    model_cfg.gate_exp_alpha,
                )
                logger.info("  gate_layer_index = %s", model_cfg.gate_layer_index)
                logger.info("  gate_location = %s", model_cfg.gate_location)

            # HF-ish: log a small markdown model card to stdout and (optionally) W&B
            model_card_md = build_model_card(cfg, world_size=world_size)
            logger.info("Model card (markdown):\n%s", model_card_md)
            if wandb_run is not None:
                try:
                    # Store in run summary for convenience
                    wandb_run.summary["model_card"] = model_card_md
                except Exception:
                    logger.warning("Failed to attach model card to W&B run.", exc_info=False)

        # ----------------------------
        # Training loop
        # ----------------------------
        for epoch in range(cfg.optim.num_epochs):
            train_sampler.set_epoch(epoch)
            epoch_loss = 0.0

            if rank == 0:
                progress_bar = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch + 1}/{cfg.optim.num_epochs}",
                    leave=True,
                )
            else:
                progress_bar = train_loader

            for step, batch in enumerate(progress_bar):
                loss, stats, images = self._forward_batch(
                    ddp_model, batch, device, cfg.optim.fp16_precision
                )

                # Gated-only debug visualizations (masks + SIM + W&B artifacts)
                if (
                    cfg.variant == "gated"
                    and cfg.debug.save_debug_masks
                    and rank == 0
                    and isinstance(stats, dict)
                    and step == 0  # first step of each epoch
                ):
                    log_debug_artifacts(
                        cfg=cfg,
                        stats=stats,
                        images=images,
                        epoch=epoch,
                        global_step=global_step,
                        model_cfg=model_cfg,
                        logger=logger,
                        wandb_run=wandb_run,
                    )

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()

                # Unscale gradients (if enabled) so grad norms are in the true scale
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)

                # Compute global norms BEFORE the optimizer step, for both variants
                grad_norm = compute_global_grad_norm(ddp_model.module)
                param_norm = compute_global_param_norm(ddp_model.module)

                scaler.step(optimizer)
                scaler.update()

                # Step LR scheduler (Hydra-configured)
                lr_scheduler.step()

                # Derive WD schedule from LR schedule
                if base_lr > 0.0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    lr_scale = float(current_lr) / float(base_lr)
                else:
                    lr_scale = 1.0

                for group in optimizer.param_groups:
                    group["weight_decay"] = float(base_weight_decay) * lr_scale

                # EMA update
                if (
                    global_step > cfg.ema.update_gamma_after_step
                    and global_step % cfg.ema.update_gamma_every_n_steps == 0
                ):
                    # Compute gamma from the EMA scheduler, then apply EMA
                    gamma = ema_scheduler(global_step)
                    ddp_model.module.update_params(gamma)

                if global_step <= cfg.ema.update_gamma_after_step:
                    ddp_model.module.copy_params()

                # Stats & logging
                loss_value = loss.item()
                total_loss += loss_value
                epoch_loss += loss_value

                avg_loss = total_loss / (global_step + 1)
                ep_loss = epoch_loss / (step + 1)

                if rank == 0:
                    # Build description + metrics in a unified way
                    desc, metrics = self._build_step_desc_and_metrics(
                        epoch=epoch,
                        num_epochs=cfg.optim.num_epochs,
                        global_step=global_step,
                        loss_value=loss_value,
                        epoch_loss_avg=ep_loss,
                        total_loss_avg=avg_loss,
                        gamma=gamma,
                        optimizer=optimizer,
                        grad_norm=grad_norm,
                        param_norm=param_norm,
                        stats=stats,
                        model_cfg=model_cfg,
                    )

                    # Show progress in-place on console
                    progress_bar.set_description(desc)
                    # Log per-step line at DEBUG level -> only file, not console
                    logger.debug(desc)

                    # Log to W&B (no-op if wandb_run is None)
                    log_metrics(wandb_run, metrics, step=global_step + 1)

                global_step += 1

                # Checkpointing & eval only on rank 0
                if rank == 0 and global_step % cfg.logging.log_every_n_steps == 0:
                    save_checkpoint(
                        ddp_model,
                        cfg.logging.save_model_dir,
                        training_ckpt_name,
                        encoder_ckpt_name,
                    )
                    logger.info(
                        "Saved checkpoints at step %d to %s",
                        global_step + 1,
                        cfg.logging.save_model_dir,
                    )

                maybe_run_eval(
                    cfg,
                    rank,
                    global_step,
                    cfg.logging.log_every_n_steps,
                    stl10_eval,
                    ddp_model,
                    logger=logger,
                )

        if rank == 0:
            logger.info("Training completed.")
            if wandb_run is not None:
                try:
                    import wandb  # type: ignore
                    wandb_run.finish()
                except Exception:
                    logger.warning("Failed to properly finish W&B run.", exc_info=False)

        dist.destroy_process_group()
