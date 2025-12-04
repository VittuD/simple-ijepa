# simple_ijepa/training/schedulers.py
"""
Learning rate & EMA schedulers in a HuggingFace-style API.

All LR schedulers in this module support an *optional linear warmup*:

    - `num_warmup_steps` (int, default = 0):
        * If 0: no warmup, the schedule starts at full LR (scale 1.0).
        * If > 0: LR scale ramps linearly from 0.0 → 1.0 over the first
          `num_warmup_steps` optimizer steps.

After warmup, each scheduler applies its own decay rule.

- LR schedulers (return torch.optim.lr_scheduler.LambdaLR):
    * "constant"   -> constant LR after warmup
    * "linear"     -> linear decay to 0 after warmup
    * "cosine"     -> cosine decay after warmup
    * "step"       -> stepwise multiplicative decay after warmup

- EMA scheduler (returns a callable step -> gamma):
    * "ema_cosine" -> cosine schedule for EMA coefficient (no warmup here;
                     EMA warmup is controlled by the training loop).
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Callable, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


LRSchedulerName = Literal["constant", "linear", "cosine", "step"]
EMASchedulerName = Literal["ema_cosine"]
SchedulerName = Literal["constant", "linear", "cosine", "step", "ema_cosine"]

SchedulerReturn = Union[LambdaLR, Callable[[int], float]]


# ---------------------------------------------------------------------------
# LR schedulers (all with optional linear warmup)
# ---------------------------------------------------------------------------

def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
) -> LambdaLR:
    """
    Constant LR after an optional linear warmup.
    """
    if num_warmup_steps < 0:
        raise ValueError(f"`num_warmup_steps` must be >= 0, got {num_warmup_steps}.")

    def lr_lambda(current_step: int) -> float:
        if num_warmup_steps == 0:
            return 1.0

        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def get_step_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    *,
    num_steps: int,
    gamma: float,
) -> LambdaLR:
    """
    Stepwise multiplicative LR decay after an optional linear warmup.
    """
    if num_training_steps <= 0:
        raise ValueError(f"`num_training_steps` must be > 0, got {num_training_steps}.")
    if num_warmup_steps < 0:
        raise ValueError(f"`num_warmup_steps` must be >= 0, got {num_warmup_steps}.")

    def lr_lambda(current_step: int) -> float:
        # Linear warmup
        if current_step < num_warmup_steps and num_warmup_steps > 0:
            return float(current_step) / float(max(1, num_warmup_steps))

        # No decay or no post-warmup region
        if num_steps <= 0:
            return 1.0

        steps_after_warmup = num_training_steps - num_warmup_steps
        if steps_after_warmup <= 0:
            return 1.0

        # Shift to post-warmup coordinates
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
    Linear LR decay to 0 after an optional linear warmup.
    """
    if num_training_steps <= 0:
        raise ValueError(f"`num_training_steps` must be > 0, got {num_training_steps}.")
    if num_warmup_steps < 0:
        raise ValueError(f"`num_warmup_steps` must be >= 0, got {num_warmup_steps}.")

    def lr_lambda(current_step: int) -> float:
        # Linear warmup
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
    Cosine LR decay after an optional linear warmup.
    """
    if num_training_steps <= 0:
        raise ValueError(f"`num_training_steps` must be > 0, got {num_training_steps}.")
    if num_warmup_steps < 0:
        raise ValueError(f"`num_warmup_steps` must be >= 0, got {num_warmup_steps}.")

    def lr_lambda(current_step: int) -> float:
        # Linear warmup
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
# EMA scheduler (internal helper, no LR warmup involved)
# ---------------------------------------------------------------------------

def _get_ema_cosine_scheduler(
    num_training_steps: int,
    base_gamma: float,
) -> Callable[[int], float]:
    """
    Cosine schedule for EMA coefficient gamma, matching the previous
    utils.update_gamma() behavior:

        tau_k = 1 - (1 - base_gamma) * (cos(pi * k / K) + 1) / 2

    EMA warmup / start behavior (e.g. "copy params until step N") is handled
    by the training loop, not by this function.
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

    LR schedulers (require `optimizer` and support optional linear warmup):
        - "constant"  -> constant LR after warmup
        - "linear"    -> linear decay to 0 after warmup
        - "cosine"    -> cosine decay after warmup
        - "step"      -> stepwise multiplicative decay after warmup

    EMA scheduler (ignores `optimizer`, returns callable step -> gamma):
        - "ema_cosine" -> cosine schedule in [base_gamma, 1.0]
    """
    # ---- LR schedulers ----
    if name == "constant":
        if optimizer is None:
            raise ValueError("`optimizer` must be provided for 'constant' scheduler.")
        return get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
        )

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
        f"Expected one of 'constant', 'linear', 'cosine', 'step', 'ema_cosine'."
    )
