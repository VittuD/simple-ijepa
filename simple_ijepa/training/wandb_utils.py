# simple_ijepa/training/wandb_utils.py

from __future__ import annotations

import os
import logging
from typing import Optional, Any

from omegaconf import OmegaConf


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

      baseline-stl10-bs256-ep100-lr3e-4
      gated-stl10-bs256-ep100-lr1e-4-lam1-a4-L2-post

    Works with both OmegaConf DictConfig and dataclass-like configs.
    """
    parts = []

    # variant: baseline / gated
    parts.append(cfg.variant)

    # dataset
    if getattr(cfg, "data", None) is not None and getattr(cfg.data, "dataset_name", None) is not None:
        parts.append(cfg.data.dataset_name)

    # core training knobs
    parts.append(f"bs{cfg.dataloader.batch_size}")
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

    slug = "-".join(str(p) for p in parts)
    return slug


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
