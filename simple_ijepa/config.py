# simple_ijepa/config.py

from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore


@dataclass
class TrainConfig:
    # which variant: "baseline" or "gated"
    variant: str = "baseline"

    # data
    dataset_path: str = "./data"
    dataset_name: str = "stl10"

    # output
    save_model_dir: str = "./models"

    # training
    num_epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    fp16_precision: bool = False

    # (kept for backwards compat even if not used)
    emb_dim: int = 768

    # logging / ckpt
    log_every_n_steps: int = 200
    ckpt_path: Optional[str] = None

    # EMA
    gamma: float = 0.996
    update_gamma_after_step: int = 1
    update_gamma_every_n_steps: int = 1

    # dataloader
    num_workers: int = 8

    # gated-only
    lambda_gates: float = 1.0
    gate_exp_alpha: float = 5.0

    # debugging / visualization
    # If True and variant == "gated", rank 0 saves mask visualizations
    # once per epoch (first step) using save_debug_masks().
    save_debug_masks: bool = False


cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)
