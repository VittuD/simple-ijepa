# simple_ijepa/config.py

from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore

from simple_ijepa.configuration_ijepa import IJEPAConfig


# ------------------------------------------------------------------
# Sub-configs
# ------------------------------------------------------------------

@dataclass
class DataConfig:
    # dataset selection
    dataset_path: str = "./data"
    dataset_name: str = "stl10"


@dataclass
class OptimConfig:
    # optimization / training loop
    num_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    fp16_precision: bool = True


@dataclass
class DataloaderConfig:
    # dataloader-related knobs
    batch_size: int = 1024
    num_workers: int = 8


@dataclass
class EMAConfig:
    # EMA teacher update
    gamma: float = 0.996
    update_gamma_after_step: int = 1
    update_gamma_every_n_steps: int = 1


@dataclass
class LoggingConfig:
    # logging / checkpointing / output dirs
    save_model_dir: str = "./models"
    log_every_n_steps: int = 98
    ckpt_path: Optional[str] = None

    # --- W&B integration ---
    # If True, rank 0 will initialize a Weights & Biases run and log metrics.
    use_wandb: bool = True

    # W&B project name (e.g. "ijepa_gate_loss").
    wandb_project: Optional[str] = "ijepa_gate_loss"

    # W&B entity (your username or team name), e.g. "vitturini-davide".
    wandb_entity: Optional[str] = "vitturini-davide"

    # Optional custom run name. If None, we auto-generate a slug.
    wandb_run_name: Optional[str] = None

    # Path to a text file that contains the W&B API key (single line).
    wandb_api_key_file: Optional[str] = ".secrets"

@dataclass
class DebugConfig:
    # debugging / visualization
    # If True and variant == "gated", rank 0 saves mask visualizations
    # once per epoch (first step) using save_debug_masks().
    save_debug_masks: bool = False


# ------------------------------------------------------------------
# Root experiment config
# ------------------------------------------------------------------

@dataclass
class TrainConfig:
    """
    Root Hydra config with structured sub-sections.
    """

    # which variant: "baseline" or "gated"
    variant: str = "baseline"

    # model architecture + gating
    model: IJEPAConfig = field(default_factory=IJEPAConfig)

    # structured sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def __post_init__(self):
        # Still safe-guard if Hydra or overrides set model=None
        if self.model is None:
            self.model = IJEPAConfig()


cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)
