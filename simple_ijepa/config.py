# simple_ijepa/config.py

from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore

from simple_ijepa.configuration_ijepa import IJEPAConfig


@dataclass
class TrainConfig:
    # which variant: "baseline" or "gated"
    variant: str = "baseline"

    # model architecture
    model: IJEPAConfig = field(default_factory=IJEPAConfig)

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
    fp16_precision: bool = True

    # logging / ckpt
    log_every_n_steps: int = 98
    ckpt_path: Optional[str] = None

    # EMA
    gamma: float = 0.996
    update_gamma_after_step: int = 1
    update_gamma_every_n_steps: int = 1

    # dataloader
    num_workers: int = 8

    # debugging / visualization
    # If True and variant == "gated", rank 0 saves mask visualizations
    # once per epoch (first step) using save_debug_masks().
    save_debug_masks: bool = False


cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)
