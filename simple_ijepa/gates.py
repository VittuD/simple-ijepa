# simple_ijepa/gates.py

import torch
from torch import nn


class HardConcreteGate(nn.Module):
    r"""
    Hard-concrete-style gate for token sparsity.

    - During training: samples continuous gate values `z ∈ (0, 1)` using a
      Concrete (relaxed Bernoulli) reparameterization.
    - During eval: uses deterministic hard gates in `{0, 1}` by thresholding
      `sigmoid(log_alpha)` at 0.5.
    - The open probability is defined as:

          p_open = sigmoid(log_alpha)

      which we use for L0-style sparsity penalties.

    This implementation is simplified compared to Louizos et al. (2017):
    we do not clamp gates to {0, 1} during training, and `gamma` / `zeta`
    are kept only for API compatibility.

    For a more detailed derivation and design notes, see `docs/gating.md`.
    """

    def __init__(
        self,
        beta: float = 2.0 / 3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
    ):
        """
        Args:
            beta:  Temperature of the Concrete distribution; smaller values
                   make gates sharper (closer to {0,1}).
            gamma, zeta: Unused in this simplified implementation, kept to
                         avoid breaking callers and for possible extensions.
        """
        super().__init__()
        self.beta = beta
        # gamma and zeta are unused, but kept to avoid breaking callers.
        self.gamma = gamma
        self.zeta = zeta

    def forward(self, log_alpha: torch.Tensor, training: bool | None = None):
        """
        Args:
            log_alpha: Tensor of shape (...,) with unconstrained log-parameters.
                       Values around 0 correspond to p_open ≈ 0.5.
            training:  Whether to use stochastic sampling (training) or
                       deterministic hard gating (eval). If None, uses
                       `self.training`.

        Returns:
            z:      Gate values, same shape as log_alpha.
                    - Training: continuous in (0, 1).
                    - Eval: 0/1 hard gates via threshold at 0.5.
            p_open: Open probability `sigmoid(log_alpha)`, same shape.
        """
        if training is None:
            training = self.training

        if training:
            # Concrete (aka Gumbel-Softmax with 2 categories) reparameterization
            u = torch.rand_like(log_alpha)
            logit_u = torch.log(u) - torch.log(1.0 - u)  # logistic noise
            s = torch.sigmoid((logit_u + log_alpha) / self.beta)
            z = s  # continuous gate in (0, 1), no clamping
        else:
            # Deterministic hard gating at eval time
            s = torch.sigmoid(log_alpha)
            z = (s > 0.5).to(log_alpha.dtype)

        p_open = self.prob_open(log_alpha)
        return z, p_open

    def prob_open(self, log_alpha: torch.Tensor) -> torch.Tensor:
        """
        Open probability used for L0-style sparsity penalty:

            p_open = sigmoid(log_alpha)

        Values:
          * p_open ≈ 0.5 when log_alpha ≈ 0,
          * p_open → 1 as log_alpha → +∞,
          * p_open → 0 as log_alpha → −∞.
        """
        return torch.sigmoid(log_alpha)
