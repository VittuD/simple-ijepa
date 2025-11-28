import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import get_original_cwd
from torch.nn.parallel import DistributedDataParallel as DDP

from simple_ijepa.config import TrainConfig
from simple_ijepa.stl10_eval import STL10Eval
from simple_ijepa.utils import GatedPredictorEncoder


def setup_distributed() -> Tuple[int, int, int]:
    """Initialize the default process group using torchrun's env variables."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def set_seed(seed: int, rank: int) -> None:
    full_seed = seed + rank
    random.seed(full_seed)
    np.random.seed(full_seed)
    torch.manual_seed(full_seed)
    torch.cuda.manual_seed_all(full_seed)


def prepare_paths(cfg: TrainConfig) -> TrainConfig:
    """
    Make dataset & model dirs relative to the original project root,
    not Hydra's per-run working directory.
    """
    root = get_original_cwd()
    if not os.path.isabs(cfg.data.dataset_path):
        cfg.data.dataset_path = os.path.join(root, cfg.data.dataset_path)
    if not os.path.isabs(cfg.logging.save_model_dir):
        cfg.logging.save_model_dir = os.path.join(root, cfg.logging.save_model_dir)
    return cfg


def save_checkpoint(
    ddp_model: DDP,
    save_dir: str,
    training_ckpt_name: str,
    encoder_ckpt_name: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, training_ckpt_name)
    enc_path = os.path.join(save_dir, encoder_ckpt_name)
    torch.save(ddp_model.module.state_dict(), ckpt_path)
    ddp_model.module.save_encoder(enc_path)


def maybe_run_eval(
    cfg: TrainConfig,
    rank: int,
    global_step: int,
    log_every_n_steps: int,
    stl10_eval: Optional[STL10Eval],
    ddp_model: DDP,
) -> None:
    """
    Run STL10 evaluation if we are on rank 0 and at the right step.
    """
    if rank != 0 or stl10_eval is None:
        return
    if global_step % log_every_n_steps != 0:
        return

    if cfg.variant == "baseline":
        # Standard path: use teacher encoder as in original code
        print("Evaluating STL10 with teacher encoder...")
        stl10_eval.evaluate(ddp_model.module)

        # Comparable alternative: use student encoder
        print("Evaluating STL10 with student encoder...")
        stl10_eval.evaluate(EncoderAsIJEPA(ddp_model.module.context_encoder))
    else:
        # Reuse STL10Eval by wrapping custom encoders
        print("Evaluating STL10 with student + learned gates...")
        gated_encoder = GatedPredictorEncoder(ddp_model.module, mode="gated").to(
            stl10_eval.device
        )
        stl10_eval.evaluate(EncoderAsIJEPA(gated_encoder))

        print("Evaluating STL10 with student + all gates open...")
        all_open_encoder = GatedPredictorEncoder(
            ddp_model.module, mode="all_open"
        ).to(stl10_eval.device)
        stl10_eval.evaluate(EncoderAsIJEPA(all_open_encoder))

    print("!" * 100)


class EncoderAsIJEPA:
    """
    Tiny shim so we can re-use STL10Eval.evaluate(), which expects
    an object with a `.target_encoder` attribute.
    """

    def __init__(self, encoder: torch.nn.Module):
        self.target_encoder = encoder
