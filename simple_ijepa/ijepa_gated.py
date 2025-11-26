# simple_ijepa/ijepa_gated.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from simple_ijepa.ijepa import BaseIJEPA
from simple_ijepa.utils import trunc_normal_
from simple_ijepa.transformer import VisionTransformer
from simple_ijepa.gates import HardConcreteGate


class GatedIJEPA(BaseIJEPA):
    r"""
    Gated variant of I-JEPA where the student encoder decides which tokens to
    use as visible context and which to mask, via differentiable hard-concrete
    gates.

    High-level idea
    ---------------
    For an image x, we compute:

      * Teacher / target encoder (EMA of the student):

          z^{(T)} = h_ξ(x) ∈ R^{N×D}

      * Student / context encoder:

          z^{(S)} = h_θ(x) ∈ R^{N×D}

      * Gating network on student tokens:

          log α = g_ϕ(z^{(S)}) ∈ R^{N}
          r ~ HardConcrete(log α) ∈ [0, 1]^N

        Each gate r_i is interpreted as:
          r_i ≈ 1   -> token i is "open" (visible context)
          r_i ≈ 0   -> token i is "closed" (masked, to be predicted)

      * Masked input to the predictor:

          c_i = r_i · z^{(S)}_i + (1 - r_i) · m

        where m ∈ R^D is a learned mask token shared across positions.

        Stack all tokens:

          C = [c_1, …, c_N]ᵀ ∈ R^{N×D}

      * Predictor:

          ẑ = p_ψ(C) ∈ R^{N×D}

    Loss
    ----
    The prediction loss is an MSE over *all* tokens (gates only affect the
    predictor input, not which tokens are evaluated):

        L_pred = mean_{b,i} || LN(ẑ_{b,i}) - LN(z^{(T)}_{b,i}) ||².

    We add a gate-usage penalty that grows exponentially with the *fraction*
    of expected open gates per sample.

    For a batch of size B with tokens i = 1..N, define:

        π_{b,i} = P(r_{b,i} > 0)           (open probability from the gate)
        f_b     = (1 / N) ∑_i π_{b,i}      (expected open fraction per sample)

        p_b     = exp(α · f_b) - 1

    Then:

        L_gates = (1 / B) ∑_b p_b

        L_total = L_pred + λ_gates · L_gates.

    Intuition:
      * L_pred encourages accurate prediction of teacher embeddings for all
        tokens given the masked context.
      * L_gates discourages using too many open gates, with a gentle penalty
        for small fractions and a sharply increasing penalty as f_b → 1.

    Observability
    -------------
    The forward pass additionally returns a dictionary of statistics:

      * "pred_loss":              L_pred (scalar)
      * "gate_penalty":           L_gates (scalar, before λ_gates)
      * "open_prob_mean":         E_{b,i}[π_{b,i}]
      * "closed_prob_mean":       1 - open_prob_mean
      * "open_frac_per_sample":   f_b  (shape: B)
      * "closed_frac_per_sample": 1 - f_b
      * "gate_map_example":       example gate map r_{b,i} reshaped to
                                  (H_p, W_p), where H_p = W_p = img_size / patch_size
      * "gate_values_full":       full gate values r_{b,i}, shape (B, N)
    """

    def __init__(
        self,
        encoder: VisionTransformer,
        hidden_emb_dim: int = 512,
        img_size: int = 96,
        patch_size: int = 8,
        predictor_depth: int = 6,
        predictor_heads: int = 6,
        lambda_gates: float = 1e-3,
        gate_beta: float = 2.0 / 3.0,
        gate_gamma: float = -0.1,
        gate_zeta: float = 1.1,
        gate_exp_alpha: float = 5.0,
    ) -> None:
        # Set up context/target encoders and EMA utilities
        super().__init__(encoder=encoder)

        self.patch_size = patch_size
        self.img_size = img_size
        self.lambda_gates = lambda_gates
        self.gate_exp_alpha = gate_exp_alpha

        # Derived: number of patches per spatial dimension
        self.num_patches_per_dim = img_size // patch_size

        # Predictor network, same style as in original I-JEPA
        self.predictor = VisionTransformer(
            image_size=img_size,
            patch_size=patch_size,
            dim=hidden_emb_dim,
            depth=predictor_depth,
            heads=predictor_heads,
            mlp_dim=hidden_emb_dim * 2,
        )

        # Shared mask token used when a gate is "closed"
        self.mask_token = nn.Parameter(torch.zeros(1, hidden_emb_dim))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)

        # Two-layer MLP with non-linearity and sigmoid to squash output
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(hidden_emb_dim),
            nn.Linear(hidden_emb_dim, hidden_emb_dim),
            nn.ReLU(),
            nn.Linear(hidden_emb_dim, 1),
        )

        # Hard-concrete gate that turns log_alpha into:
        #   r_{b,i} ∈ [0, 1] and
        #   π_{b,i} ≈ P(r_{b,i} > 0)
        self.gate = HardConcreteGate(
            beta=gate_beta,
            gamma=gate_gamma,
            zeta=gate_zeta,
        )

    def forward(
        self,
        img: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            img: Input batch of images, shape (B, C, H, W).

        Returns:
            total_loss: scalar tensor, the gated I-JEPA loss:
                        L_total = L_pred + λ_gates · L_gates

            stats: dict with monitoring information, containing:
                * "pred_loss"
                * "gate_penalty"
                * "open_prob_mean"
                * "closed_prob_mean"
                * "open_frac_per_sample"
                * "closed_frac_per_sample"
                * "gate_map_example"
                * "gate_values_full"
        """
        device = img.device

        # -------------------------
        # 1. Teacher / target path
        # -------------------------
        # Uses EMA target encoder + LayerNorm from BaseIJEPA._encode_teacher
        with torch.no_grad():
            target_features = self._encode_teacher(img)  # (B, N, D)

        B, N, D = target_features.shape

        # -------------------------
        # 2. Student embeddings
        # -------------------------
        # context_encoder processes the full image, no manual masking here.
        student_tokens = self.context_encoder(img)  # (B, N, D)

        # -------------------------
        # 3. Gating over tokens
        # -------------------------
        # gate_mlp produces a scalar log_alpha per token:
        #   log α_{b,i} ∈ R
        log_alpha = self.gate_mlp(student_tokens).squeeze(-1)  # (B, N)
        # Clamp for numerical stability
        log_alpha = torch.clamp(log_alpha, min=-10.0, max=10.0)

        # r ∈ [0, 1] are the actual gate samples (differentiable during training);
        # gate_probs ∈ (0,1) are the open probabilities P(r > 0) used for
        # the gate usage penalty.
        gate_values, gate_probs = self.gate(
            log_alpha,
            training=self.training,
        )  # (B, N), (B, N)

        # -------------------------
        # 4. Build masked input C
        # -------------------------
        # For each token i:
        #   c_i = r_i * z^{(S)}_i + (1 - r_i) * m
        # where m is a learned mask token shared across positions.
        mask_token = self.mask_token.to(device)          # (1, D)
        mask_tokens = mask_token.expand(B, N, D)         # (B, N, D)

        r = gate_values.unsqueeze(-1)                    # (B, N, 1)
        one_minus_r = 1.0 - r                            # (B, N, 1)

        context_tokens = r * student_tokens + one_minus_r * mask_tokens
        # context_tokens: (B, N, D)

        # -------------------------
        # 5. Predictor
        # -------------------------
        # Predictor operates on tokens directly (no patchify, no extra pos emb).
        predictor_output = self.predictor(
            context_tokens,
            patchify=False,
            pos_embed=False,
        )
        predictor_output = F.layer_norm(
            predictor_output,
            (predictor_output.size(-1),),
        )

        # -------------------------
        # 6. Prediction loss on all tokens
        # -------------------------
        # Gates only affect the *input* to the predictor (context_tokens).
        # Every token's prediction error contributes to the loss.
        diff = predictor_output - target_features         # (B, N, D)
        sq_norm = (diff ** 2).sum(dim=-1)                 # (B, N)

        pred_loss = sq_norm.mean()                        # scalar

        # -------------------------
        # 7. Gate sparsity penalty (normalized + exponential)
        # -------------------------
        # gate_probs_{b,i} ≈ P(r_{b,i} > 0) => "open gate" probability.
        #
        # Per-sample expected fraction of open gates:
        #
        #   f_b = (1 / N) * ∑_i gate_probs_{b,i} ∈ [0, 1]
        #
        # Exponential penalty:
        #
        #   p_b = exp(α · f_b) - 1
        #
        # L_gates = (1 / B) ∑_b p_b
        # L_total = L_pred + λ_gates · L_gates
        open_frac_per_sample = gate_probs.mean(dim=1)         # (B,)
        gate_penalty_per_sample = torch.exp(
            self.gate_exp_alpha * open_frac_per_sample
        ) - 1.0                                               # (B,)
        gate_penalty = gate_penalty_per_sample.mean()         # scalar

        total_loss = pred_loss + self.lambda_gates * gate_penalty

        # -------------------------
        # 8. Observability stats
        # -------------------------
        # Mean open probability over all tokens in batch
        open_prob_mean = gate_probs.mean()                     # scalar
        closed_prob_mean = 1.0 - open_prob_mean

        # Per-sample expected open / closed fractions
        open_frac_per_sample = gate_probs.mean(dim=1)          # (B,)
        closed_frac_per_sample = 1.0 - open_frac_per_sample    # (B,)

        # Example gate map (for logging / visualization): choose a
        # representative sample (e.g. index 0) and reshape to
        # (H_p, W_p) with H_p * W_p = N.
        H_p = self.num_patches_per_dim
        W_p = self.num_patches_per_dim
        if H_p * W_p != N:
            # Fallback: just keep a 1D vector if shape mismatch
            gate_map_example = gate_values[0].detach()
        else:
            gate_map_example = gate_values[0].detach().view(H_p, W_p)

        stats = {
            "pred_loss": pred_loss.detach(),
            "gate_penalty": gate_penalty.detach(),
            "open_prob_mean": open_prob_mean.detach(),
            "closed_prob_mean": closed_prob_mean.detach(),
            "open_frac_per_sample": open_frac_per_sample.detach(),
            "closed_frac_per_sample": closed_frac_per_sample.detach(),
            "gate_map_example": gate_map_example,         # (H_p, W_p) or (N,)
            # Full gate values for the whole batch (B, N)
            "gate_values_full": gate_values.detach(),
        }

        return total_loss, stats
