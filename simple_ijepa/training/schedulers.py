# simple_ijepa/training/schedulers.py
"""
Learning rate & EMA schedulers in a HuggingFace-style API.

- LR schedulers (return torch.optim.lr_scheduler.LambdaLR):
    * "linear"
    * "cosine"
    * "step"

- EMA scheduler (returns a callable step -> gamma):
    * "ema_cosine"
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Callable, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


LRSchedulerName = Literal["linear", "cosine", "step"]
EMASchedulerName = Literal["ema_cosine"]
SchedulerName = Literal["linear", "cosine", "step", "ema_cosine"]

SchedulerReturn = Union[LambdaLR, Callable[[int], float]]


# ---------------------------------------------------------------------------
# LR schedulers with warmup
# ---------------------------------------------------------------------------

def get_step_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    *,
    num_steps: int,
    gamma: float,
) -> LambdaLR:
    """
    Step decay scheduler with a warmup phase.

    Args:
        optimizer:         Wrapped optimizer.
        num_warmup_steps:  Warmup steps (linear 0 -> 1).
        num_training_steps:Total training steps (including warmup).
        num_steps:         Number of decay events after warmup.
        gamma:             Multiplicative factor at each decay event.
    """
    if num_training_steps <= 0:
        raise ValueError(f"`num_training_steps` must be > 0, got {num_training_steps}.")
    if num_warmup_steps < 0:
        raise ValueError(f"`num_warmup_steps` must be >= 0, got {num_warmup_steps}.")

    def lr_lambda(current_step: int) -> float:
        # Warmup
        if current_step < num_warmup_steps and num_warmup_steps > 0:
            return float(current_step) / float(max(1, num_warmup_steps))

        # No decay or no post-warmup region
        if num_steps <= 0:
            return 1.0

        steps_after_warmup = num_training_steps - num_warmup_steps
        if steps_after_warmup <= 0:
            return 1.0

        # Shift to post-warmup coords
        t = max(0, current_step - num_warmup_steps)

        step_width = steps_after_warmup / float(num_steps)
        decay_index = int(t / max(1.0, step_width))
        decay_index = max(0, min(decay_index, num_steps))

        return float(gamma) ** float(decay_index)

    return LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """
    Classic linear decay with warmup.
    """
    if num_training_steps <= 0:
        raise ValueError(f"`num_training_steps` must be > 0, got {num_training_steps}.")
    if num_warmup_steps < 0:
        raise ValueError(f"`num_warmup_steps` must be >= 0, got {num_warmup_steps}.")

    def lr_lambda(current_step: int) -> float:
        # Warmup
        if current_step < num_warmup_steps and num_warmup_steps > 0:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Linear decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
) -> LambdaLR:
    """
    Cosine annealing schedule with warmup.
    """
    if num_training_steps <= 0:
        raise ValueError(f"`num_training_steps` must be > 0, got {num_training_steps}.")
    if num_warmup_steps < 0:
        raise ValueError(f"`num_warmup_steps` must be >= 0, got {num_warmup_steps}.")

    def lr_lambda(current_step: int) -> float:
        # Warmup
        if current_step < num_warmup_steps and num_warmup_steps > 0:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay
        if current_step >= num_training_steps:
            return 0.0

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_term = math.cos(math.pi * 2.0 * num_cycles * progress)
        return 0.5 * (1.0 + cosine_term)

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# EMA scheduler (internal helper)
# ---------------------------------------------------------------------------

def _get_ema_cosine_scheduler(
    num_training_steps: int,
    base_gamma: float,
) -> Callable[[int], float]:
    """
    Cosine schedule for EMA coefficient gamma, matching utils.update_gamma:

        tau_k = 1 - (1 - base_gamma) * (cos(pi * k / K) + 1) / 2
    """
    if num_training_steps <= 0:
        raise ValueError(f"`num_training_steps` must be > 0, got {num_training_steps}.")

    K = float(num_training_steps)
    base = float(base_gamma)

    def fn(current_step: int) -> float:
        k = float(current_step)
        tau = 1.0 - (1.0 - base) * (math.cos(math.pi * k / K) + 1.0) / 2.0
        return float(tau)

    return fn


# ---------------------------------------------------------------------------
# Unified factory
# ---------------------------------------------------------------------------

def get_scheduler(
    name: SchedulerName,
    optimizer: Optional[Optimizer] = None,
    num_warmup_steps: int = 0,
    num_training_steps: int = 0,
    *,
    num_steps: Optional[int] = None,
    gamma: Optional[float] = None,
    num_cycles: float = 0.5,
    ema_base_gamma: Optional[float] = None,
) -> SchedulerReturn:
    """
    Generic factory for LR and EMA schedulers.

    LR schedulers (require `optimizer`):
        - "linear"
        - "cosine"
        - "step"

    EMA scheduler (ignores `optimizer`, returns callable step -> gamma):
        - "ema_cosine"

    Returns:
        - For LR: LambdaLR
        - For EMA: Callable[[int], float]
    """
    # ---- LR schedulers ----
    if name == "linear":
        if optimizer is None:
            raise ValueError("`optimizer` must be provided for 'linear' scheduler.")
        return get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    if name == "cosine":
        if optimizer is None:
            raise ValueError("`optimizer` must be provided for 'cosine' scheduler.")
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
        )

    if name == "step":
        if optimizer is None:
            raise ValueError("`optimizer` must be provided for 'step' scheduler.")
        if num_steps is None:
            raise ValueError("`num_steps` must be provided for 'step' scheduler.")
        if gamma is None:
            raise ValueError("`gamma` must be provided for 'step' scheduler.")

        return get_step_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_steps=num_steps,
            gamma=gamma,
        )

    # ---- EMA scheduler ----
    if name == "ema_cosine":
        if ema_base_gamma is None:
            raise ValueError("`ema_base_gamma` must be provided for 'ema_cosine' scheduler.")
        return _get_ema_cosine_scheduler(
            num_training_steps=num_training_steps,
            base_gamma=ema_base_gamma,
        )

    raise ValueError(
        f"Unknown scheduler name: {name!r}. "
        f"Expected one of 'linear', 'cosine', 'step', 'ema_cosine'."
    )
