# simple_ijepa/model_card.py

from __future__ import annotations

from simple_ijepa.config import TrainConfig


def _fmt_bool(x: bool) -> str:
    return "yes" if bool(x) else "no"


def build_model_card(cfg: TrainConfig, world_size: int = 1) -> str:
    """
    Return a short markdown model card-style summary for the current run.

    This is intentionally lightweight: it is meant for logging (stdout,
    W&B summary, etc.), not for writing to disk as a full formal card.
    """
    gbs = cfg.dataloader.batch_size * max(1, int(world_size))

    m = cfg.model
    lines: list[str] = []

    # Title
    lines.append(f"# Simple I-JEPA ({cfg.variant})")
    lines.append("")
    lines.append("Auto-generated run summary. This block is safe to paste in a README or W&B notes.")
    lines.append("")

    # Dataset / task
    lines.append("## Data")
    lines.append("")
    lines.append(f"- **Dataset**: `{cfg.data.dataset_name}`")
    lines.append(f"- **Dataset path**: `{cfg.data.dataset_path}`")
    lines.append("")

    # Model
    lines.append("## Model")
    lines.append("")
    lines.append(f"- **Backbone**: VisionTransformer")
    lines.append(f"- **Image size**: {m.image_size}")
    lines.append(f"- **Patch size**: {m.patch_size}")
    lines.append(f"- **Embedding dim**: {m.hidden_dim}")
    lines.append(f"- **Depth / heads**: {m.depth} / {m.heads}")
    lines.append(f"- **Predictor depth / heads**: {m.predictor_depth} / {m.predictor_heads}")
    lines.append(f"- **Num targets**: {m.num_targets}")
    lines.append("")

    # Gated bits (if relevant)
    if cfg.variant == "gated":
        lines.append("### Gating")
        lines.append("")
        lines.append(f"- `lambda_gates`: {m.lambda_gates}")
        lines.append(f"- `gate_exp_alpha`: {m.gate_exp_alpha}")
        lines.append(f"- `gate_layer_index`: {m.gate_layer_index}")
        lines.append(f"- `gate_location`: `{m.gate_location}`")
        lines.append(f"- `gate_beta`: {m.gate_beta}")
        lines.append(f"- `gate_gamma`: {m.gate_gamma}")
        lines.append(f"- `gate_zeta`: {m.gate_zeta}")
        lines.append("")

    # Optimization
    lines.append("## Optimization")
    lines.append("")
    lines.append(f"- **Epochs**: {cfg.optim.num_epochs}")
    lines.append(f"- **Learning rate**: {cfg.optim.learning_rate}")
    lines.append(f"- **Weight decay**: {cfg.optim.weight_decay}")
    lines.append(f"- **Mixed precision (fp16)**: {_fmt_bool(cfg.optim.fp16_precision)}")
    lines.append("")
    lines.append(f"- **Per-device batch size**: {cfg.dataloader.batch_size}")
    lines.append(f"- **World size**: {world_size}")
    lines.append(f"- **Effective global batch size**: {gbs}")
    lines.append("")

    # EMA
    lines.append("## EMA Teacher")
    lines.append("")
    lines.append(f"- `gamma` (initial): {cfg.ema.gamma}")
    lines.append(f"- `update_gamma_after_step`: {cfg.ema.update_gamma_after_step}")
    lines.append(f"- `update_gamma_every_n_steps`: {cfg.ema.update_gamma_every_n_steps}")
    lines.append("")

    # Logging
    lines.append("## Logging")
    lines.append("")
    lines.append(f"- **Save dir**: `{cfg.logging.save_model_dir}`")
    lines.append(f"- **Log every n steps**: {cfg.logging.log_every_n_steps}")
    lines.append(f"- **Use W&B**: {_fmt_bool(cfg.logging.use_wandb)}")
    if cfg.logging.use_wandb:
        lines.append(f"- **W&B project**: `{cfg.logging.wandb_project}`")
        lines.append(f"- **W&B entity**: `{cfg.logging.wandb_entity}`")
    lines.append("")

    return "\n".join(lines)
