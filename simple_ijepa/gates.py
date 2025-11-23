# simple_ijepa/gates.py

import torch
from torch import nn


class HardConcreteGate(nn.Module):
    r"""
    Concrete-style gate for L0-like sparsity over tokens.

    This is a simplified variant inspired by the hard-concrete gate from:

        "Learning Sparse Neural Networks through L_0 Regularization"
        (Louizos et al., 2017),

    but with two important changes for this project:

      1. No clamping: during training, gate values z_i are continuous in (0, 1),
         not hard-clipped to exactly 0 or 1. This preserves gradients even when
         log-parameters become large in magnitude.

      2. Symmetric prior: we choose the open probability as

            p_i = sigmoid(log_alpha_i),

         so that when log_alpha_i ≈ 0, we have p_i ≈ 0.5. Under a roughly
         zero-mean initialization of log_alpha, this makes *half* the gates
         open on average and *half* closed.

    Reparameterization (training)
    -----------------------------
    For each location i, we have a log-parameter `log_alpha_i`. We define a
    continuous random variable z_i in (0, 1) via the Concrete distribution:

        u_i ~ Uniform(0, 1)
        l_i = log u_i - log(1 - u_i)        # logistic noise
        s_i = sigmoid((l_i + log_alpha_i) / beta)
        z_i = s_i

    where:
      * beta \in (0, +\infty) is a temperature (lower beta -> sharper gates).

    At evaluation time, we use a hard threshold:

        s_i = sigmoid(log_alpha_i)
        z_i = 1 if s_i > 0.5 else 0.

    Probability of being "open"
    ---------------------------
    We define the (differentiable) open probability as:

        p_i = sigmoid(log_alpha_i),

    so that:

        E[# open gates] ≈ sum_i p_i,

    and

        E[fraction of open gates] ≈ (1 / N) * sum_i p_i.

    These expectations are what we plug into the sparsity penalty in
    `GatedIJEPA`, e.g., via an exponential function of the *fraction* of
    open gates.
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
            gamma, zeta: Kept for backward compatibility with the original
                         hard-concrete signature, but not used in this
                         simplified implementation.
        """
        super().__init__()
        self.beta = beta
        # gamma and zeta are unused, but kept to avoid breaking callers.
        self.gamma = gamma
        self.zeta = zeta

    def forward(self, log_alpha: torch.Tensor, training: bool | None = None):
        r"""
        Args:
            log_alpha: Tensor of shape (...,) - unconstrained log-parameters.
                       Values around 0 correspond to p_open ≈ 0.5.
            training:  Whether to use stochastic sampling (training) or
                       deterministic hard gating (eval). If None, uses
                       `self.training`.

        Returns:
            z:        Gate values, same shape as log_alpha.
                      - During training: continuous in (0, 1), sampled via
                        the Concrete reparameterization.
                      - During eval: hard 0/1 via threshold at 0.5.
            p_open:   Tensor of same shape - probability that each gate is
                      "open", defined as sigmoid(log_alpha).
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
        r"""
        Open probability used for L0-style sparsity penalty:

            p_i = sigmoid(log_alpha_i).

        This choice ensures:
          * p_i ≈ 0.5 when log_alpha_i ≈ 0,
          * p_i -> 1 as log_alpha_i -> +∞,
          * p_i -> 0 as log_alpha_i -> -∞.

        Under an approximately zero-mean initialization of log_alpha, this
        yields ~50% open and ~50% closed gates on average at initialization.

        Args:
            log_alpha: Tensor of shape (...,)

        Returns:
            p_open: Same shape as log_alpha; each entry in (0, 1).
        """
        return torch.sigmoid(log_alpha)
