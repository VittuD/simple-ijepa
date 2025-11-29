# simple_ijepa/simple_ijepa/training/ddp_trainer.py

import os
from typing import Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
from torchvision.datasets import STL10

from simple_ijepa.transformer import VisionTransformer
from simple_ijepa.ijepa import IJEPA
from simple_ijepa.ijepa_gated import GatedIJEPA
from simple_ijepa.stl10_eval import STL10Eval
from simple_ijepa.utils import (
    update_gamma,
    training_transforms,
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


SEED = 42


class IJEPATrainerDDP:
    def __init__(self, cfg: TrainConfig):
        self.cfg = prepare_paths(cfg)

    def _build_model_and_optimizer(
        self,
        device: torch.device,
        local_rank: int,
        rank: int,
        model_cfg: IJEPAConfig,
        logger,
    ) -> Tuple[torch.nn.Module, torch.optim.Optimizer, str, str]:
        """
        Build the encoder + IJEPA or GatedIJEPA model, load an optional checkpoint,
        and create the optimizer.

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
        """
        Create STL10-based training dataset and DDP-aware DataLoader.
        """
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
        )
        if cfg.variant == "baseline":
            loader_kwargs["collate_fn"] = collate_fn

        train_loader = DataLoader(**loader_kwargs)
        return train_loader, train_sampler

    def _forward_batch(
        self,
        ddp_model: DDP,
        batch,
        device: torch.device,
        fp16: bool,
    ):
        """
        Run a forward pass for a batch, handling the differences between
        baseline and gated variants. Returns (loss, stats, images).
        """
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

        total_num_steps = (len(train_loader) * (cfg.optim.num_epochs + 2)) - cfg.ema.update_gamma_after_step
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

                # Gated-only debug visualizations (masks + SSIM + W&B artifacts)
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
                scaler.step(optimizer)
                scaler.update()

                # EMA update
                if (
                    global_step > cfg.ema.update_gamma_after_step
                    and global_step % cfg.ema.update_gamma_every_n_steps == 0
                ):
                    ddp_model.module.update_params(gamma)
                    gamma = update_gamma(global_step, total_num_steps, cfg.ema.gamma)

                if global_step <= cfg.ema.update_gamma_after_step:
                    ddp_model.module.copy_params()

                # Stats & logging
                loss_value = loss.item()
                total_loss += loss_value
                epoch_loss += loss_value

                avg_loss = total_loss / (global_step + 1)
                ep_loss = epoch_loss / (step + 1)

                if rank == 0:
                    current_lr = optimizer.param_groups[0]["lr"]

                    if cfg.variant == "baseline":
                        desc = (
                            f"Epoch {epoch+1}/{cfg.optim.num_epochs} | "
                            f"Step {global_step+1} | "
                            f"Epoch Loss: {ep_loss:.7f} | "
                            f"Total Loss: {avg_loss:.7f} | "
                            f"EMA gamma: {gamma:.6f} | "
                            f"Lr: {current_lr:.6f}"
                        )

                        metrics = {
                            "train/epoch": epoch + 1,
                            "train/global_step": global_step + 1,
                            "train/loss_step": loss_value,
                            "train/loss_epoch": ep_loss,
                            "train/loss_avg": avg_loss,
                            "train/ema_gamma": gamma,
                            "train/lr": current_lr,
                        }
                    else:
                        pred_loss = stats["pred_loss"].item()
                        gate_penalty = stats["gate_penalty"].item()
                        open_prob_mean = stats["open_prob_mean"].item()
                        closed_prob_mean = stats["closed_prob_mean"].item()

                        desc = (
                            f"Epoch {epoch+1}/{cfg.optim.num_epochs} | "
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

                        metrics = {
                            "train/epoch": epoch + 1,
                            "train/global_step": global_step + 1,
                            "train/loss_step": loss_value,
                            "train/loss_epoch": ep_loss,
                            "train/loss_avg": avg_loss,
                            "train/ema_gamma": gamma,
                            "train/lr": current_lr,
                            "gates/pred_loss": pred_loss,
                            "gates/gate_penalty": gate_penalty,
                            "gates/open_prob_mean": open_prob_mean,
                            "gates/closed_prob_mean": closed_prob_mean,
                        }

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
