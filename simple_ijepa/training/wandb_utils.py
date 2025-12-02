# simple_ijepa/training/wandb_utils.py

from __future__ import annotations

import os
from datetime import datetime
import logging
from typing import Optional, Any
import torch

from omegaconf import OmegaConf

from simple_ijepa.utils import (
    save_debug_masks,
    save_sim_heatmap_grid,
    cosine_sim_matrix,
)


def _load_api_key_from_file(
    path: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Read a W&B API key from a plain text file and put it into WANDB_API_KEY
    (if not already set). The file should contain the key on a single line.
    """
    if not path:
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            key = f.read().strip()
            os.environ["WANDB_API_KEY"] = key
    except FileNotFoundError:
        if logger is not None:
            logger.warning("wandb_api_key_file '%s' not found.", path)
        return
    except Exception as e:
        if logger is not None:
            logger.warning(
                "Failed to read wandb_api_key_file '%s': %s",
                path,
                e,
            )
        return

    if not key:
        if logger is not None:
            logger.warning("wandb_api_key_file '%s' is empty.", path)
        return

    # Respect an already-set API key if present
    if "WANDB_API_KEY" not in os.environ:
        os.environ["WANDB_API_KEY"] = key


def _short_float(x: float) -> str:
    """
    Compact float string for slugs:
      3e-4  -> '3e-4'
      0.001 -> '0.001'
      1.0   -> '1'
    """
    xf = float(x)
    if xf.is_integer():
        return str(int(xf))
    # 3 significant digits is usually enough for lr / lambda, etc.
    return f"{xf:.3g}"


def make_run_slug(cfg: Any) -> str:
    """
    Build a slug-style run name with shorthand parameters, e.g.:

      baseline-bs256_gbs1024-ep100-lr3e-4-stl10
      gated-bs256_gbs2048-ep100-lr1e-4-lam1-a4-L2-post-stl10

    Works with both OmegaConf DictConfig and dataclass-like configs.
    """
    parts = []

    # variant: baseline / gated
    parts.append(cfg.variant)

    # per-device and global batch size
    per_device_bs = cfg.dataloader.batch_size
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        world_size = 1  # fallback if env var is weird / missing

    effective_global_bs = per_device_bs * world_size

    # this is the bit you wanted
    parts.append(f"bs{per_device_bs}_gbs{effective_global_bs}")

    # core training knobs
    parts.append(f"ep{cfg.optim.num_epochs}")
    parts.append(f"lr{_short_float(cfg.optim.learning_rate)}")

    # gated-specific knobs
    if cfg.variant == "gated":
        m = cfg.model
        parts.append(f"lam{_short_float(m.lambda_gates)}")
        parts.append(f"a{_short_float(m.gate_exp_alpha)}")
        layer = "final" if m.gate_layer_index is None else str(m.gate_layer_index)
        parts.append(f"L{layer}")
        parts.append(m.gate_location)

    # dataset (optional, if present)
    if getattr(cfg, "data", None) is not None and getattr(cfg.data, "dataset_name", None) is not None:
        parts.append(cfg.data.dataset_name)

    slug = "-".join(str(p) for p in parts)
    return slug


def make_timed_run_slug(cfg: Any) -> str:
    """
    Like make_run_slug, but appends a timestamp so each run is unique:

      baseline-stl10-bs256-ep100-lr3e-4-...-20251129-143215
    """
    base = make_run_slug(cfg)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{base}-{ts}"


def make_run_tags(cfg: Any, world_size: int) -> list[str]:
    """
    Build a list of human-readable tags that mirror the slug, but with
    full-length key=value text for easier filtering in the W&B UI.
    """
    tags: list[str] = []

    # Generic / shared
    tags.append(f"variant={cfg.variant}")

    if getattr(cfg, "data", None) is not None and getattr(cfg.data, "dataset_name", None) is not None:
        tags.append(f"dataset={cfg.data.dataset_name}")

    tags.append(f"batch_size={cfg.dataloader.batch_size}")
    tags.append(f"num_epochs={cfg.optim.num_epochs}")
    tags.append(f"learning_rate={cfg.optim.learning_rate}")
    tags.append(f"weight_decay={cfg.optim.weight_decay}")
    tags.append(f"fp16_precision={cfg.optim.fp16_precision}")
    tags.append(f"world_size={world_size}")

    effective_global_batch_size = cfg.dataloader.batch_size * world_size
    tags.append(f"effective_global_batch_size={effective_global_batch_size}")

    # Model architecture
    m = cfg.model
    tags.append(f"image_size={m.image_size}")
    tags.append(f"patch_size={m.patch_size}")
    tags.append(f"hidden_dim={m.hidden_dim}")
    tags.append(f"depth={m.depth}")
    tags.append(f"heads={m.heads}")
    tags.append(f"predictor_depth={m.predictor_depth}")
    tags.append(f"predictor_heads={m.predictor_heads}")
    tags.append(f"num_targets={m.num_targets}")

    # Gating-specific tags
    if cfg.variant == "gated":
        tags.append(f"lambda_gates={m.lambda_gates}")
        tags.append(f"gate_exp_alpha={m.gate_exp_alpha}")
        tags.append(f"gate_layer_index={m.gate_layer_index}")
        tags.append(f"gate_location={m.gate_location}")
        tags.append(f"gate_beta={m.gate_beta}")
        tags.append(f"gate_gamma={m.gate_gamma}")
        tags.append(f"gate_zeta={m.gate_zeta}")

    return tags


def init_wandb(
    cfg: Any,
    rank: int,
    world_size: int,
    logger: Optional[logging.Logger] = None,
) -> Optional[Any]:
    """
    Initialize a W&B run on rank 0 if logging.use_wandb is True.
    Returns the run object, or None if W&B is disabled or unavailable.

    `cfg` can be an OmegaConf DictConfig or a dataclass-like object; we
    convert it to a plain dict via OmegaConf.
    """
    # Only rank 0 logs to W&B
    if not getattr(cfg.logging, "use_wandb", False) or rank != 0:
        return None

    try:
        import wandb  # type: ignore
    except ImportError:
        if logger is not None:
            logger.warning(
                "Weights & Biases (wandb) is not installed. "
                "Install it with `pip install wandb` or disable it via "
                "`logging.use_wandb=false`."
            )
        return None

    # Optional API key from file
    api_key_file = getattr(cfg.logging, "wandb_api_key_file", None)
    if api_key_file:
        _load_api_key_from_file(api_key_file, logger=logger)

    project = getattr(cfg.logging, "wandb_project", None) or "simple-ijepa"
    entity = getattr(cfg.logging, "wandb_entity", None)  # e.g. "vitturini-davide"
    run_name_override = getattr(cfg.logging, "wandb_run_name", None)
    run_name = run_name_override or make_run_slug(cfg)

    # Build tags that mirror the slug but are full text
    run_tags = make_run_tags(cfg, world_size=world_size)

    # Convert the config (DictConfig or dataclass) to a nested dict for W&B
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Add some extra distributed metadata
    if isinstance(cfg_dict, dict):
        cfg_dict.setdefault("distributed", {})
        cfg_dict["distributed"]["world_size"] = world_size
        cfg_dict["distributed"]["effective_global_batch_size"] = (
            cfg.dataloader.batch_size * world_size
        )

    wandb.login(key=os.environ.get("WANDB_API_KEY", ""))

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=cfg_dict,
        tags=run_tags,
    )

    if logger is not None:
        logger.info(
            "Initialized Weights & Biases run: entity=%s, project=%s, name=%s, tags=%s",
            entity,
            project,
            run.name,
            ", ".join(run_tags),
        )

    return run


def log_metrics(
    run: Optional[Any],
    metrics: dict,
    step: Optional[int] = None,
) -> None:
    """
    Thin wrapper around wandb.Run.log that no-ops if run is None and
    never raises inside the training loop.
    """
    if run is None:
        return

    try:
        run.log(metrics, step=step)
    except Exception:
        # Avoid breaking the training loop on W&B hiccups
        pass


def log_artifact_files(
    run: Optional[Any],
    file_paths: list[str],
    artifact_name: str,
    artifact_type: str = "debug",
    metadata: Optional[dict] = None,
) -> None:
    """
    Attach a set of files to a W&B run as an artifact.

    Args:
        run: W&B run object (or None -> no-op).
        file_paths: List of filesystem paths to attach.
        artifact_name: Name of the artifact (e.g. "debug_masks_e001_s000123").
        artifact_type: Artifact type (e.g. "debug_masks", "debug_sim").
        metadata: Optional dict stored with the artifact (epoch, step, etc.).
    """
    if run is None:
        return

    try:
        import wandb  # type: ignore

        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            metadata=metadata or {},
        )
        for p in file_paths:
            if os.path.exists(p):
                artifact.add_file(p)

        run.log_artifact(artifact)
    except Exception:
        # Never break training on W&B issues
        return

def log_debug_artifacts(
    *,
    cfg: Any,
    stats: dict,
    images: torch.Tensor,
    epoch: int,
    global_step: int,
    model_cfg: Any,
    logger: Optional[logging.Logger],
    wandb_run: Optional[Any],
) -> None:
    """
    Save mask + sim debug artifacts to disk and (optionally) log them to W&B
    as plain images at the current epoch/step.

    Called from the trainer only for:
      - gated variant
      - rank 0
      - first step of each epoch
    """
    if logger is None:
        logger = logging.getLogger("simple_ijepa.debug")

    step_str = f"e{epoch+1:03d}_s{global_step+1:06d}"
    step = global_step + 1

    # Try to import wandb only if we actually have a run
    wandb = None
    if wandb_run is not None:
        try:
            import wandb as _wandb  # type: ignore
            wandb = _wandb
        except Exception:
            wandb = None

    # -------------------
    # 1) Patch masks (single combined image + .pt with tensors)
    # -------------------
    if "gate_values_full" in stats:
        try:
            gate_values_full = stats["gate_values_full"]
            # Save PNG grid
            combined_path = save_debug_masks(
                images=images,
                gate_values_full=gate_values_full,
                epoch=epoch,
                global_step=global_step,
                save_root=cfg.logging.save_model_dir,
                image_size=model_cfg.image_size,
                patch_size=model_cfg.patch_size,
                max_images=8,
            )
            logger.info(
                "Saved combined debug masks for epoch %d, step %d under %s/debug_masks",
                epoch + 1,
                global_step + 1,
                cfg.logging.save_model_dir,
            )

            # Save .pt with the raw tensors used for the mask visualization
            # (slice to match max_images)
            max_images = 8
            B = images.shape[0]
            if isinstance(gate_values_full, torch.Tensor):
                G = gate_values_full.shape[0]
            else:
                G = B  # fallback, should not really happen

            num_to_save = max(0, min(max_images, B, G))
            if num_to_save > 0:
                mask_pt_path = os.path.splitext(combined_path)[0] + ".pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "global_step": step,
                        "images": images[:num_to_save].detach().cpu(),
                        "gate_values_full": gate_values_full[:num_to_save].detach().cpu()
                        if isinstance(gate_values_full, torch.Tensor)
                        else gate_values_full,
                        "image_size": model_cfg.image_size,
                        "patch_size": model_cfg.patch_size,
                    },
                    mask_pt_path,
                )
                logger.info(
                    "Saved mask debug tensors (.pt) for epoch %d, step %d to %s",
                    epoch + 1,
                    global_step + 1,
                    mask_pt_path,
                )

            if wandb is not None:
                wandb_run.log(
                    {
                        "debug/masks_orig_masked": wandb.Image(combined_path),
                    },
                    step=step,
                )
        except Exception as e:
            logger.warning("Failed to save/log debug masks: %s", e)

    # -------------------
    # 2) Token sim heatmap grid (PNG + .pt with tokens)
    # -------------------
    if "gate_mlp_input_examples" in stats:
        try:
            from simple_ijepa.utils import cosine_sim_matrix, save_sim_heatmap_grid

            tokens_batch = stats["gate_mlp_input_examples"]  # (K, N, D)
            debug_dir = os.path.join(cfg.logging.save_model_dir, "debug_ssim")
            os.makedirs(debug_dir, exist_ok=True)

            ssim_png_path = os.path.join(
                debug_dir, f"{step_str}_token_metric_grid.png"
            )

            # cosine similarity grid
            save_sim_heatmap_grid(
                tokens_batch,
                ssim_png_path,
                metric_fn=cosine_sim_matrix,
            )

            logger.info(
                "Saved token similarity grid for epoch %d, step %d under %s",
                epoch + 1,
                global_step + 1,
                debug_dir,
            )

            # Save .pt with the raw tokens used to compute these grids
            if isinstance(tokens_batch, torch.Tensor):
                ssim_pt_path = os.path.splitext(ssim_png_path)[0] + ".pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "global_step": step,
                        "tokens_batch": tokens_batch.detach().cpu(),  # (K, N, D)
                    },
                    ssim_pt_path,
                )
                logger.info(
                    "Saved token similarity debug tensors (.pt) for epoch %d, step %d to %s",
                    epoch + 1,
                    global_step + 1,
                    ssim_pt_path,
                )

            if wandb is not None:
                wandb_run.log(
                    {
                        "debug/token_sim_grid": wandb.Image(ssim_png_path),
                    },
                    step=step,
                )
        except Exception as e:
            logger.warning("Failed to compute/save/log token similarity grid: %s", e)
