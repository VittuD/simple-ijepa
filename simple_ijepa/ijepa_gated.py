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

      * Gating network on (intermediate or final) student tokens:

          log α = g_ϕ(z^{(S)}) ∈ R^{N}
          r ~ HardConcrete(log α) ∈ [0, 1]^N

        Each gate r_i is interpreted as:
          r_i ≈ 1   -> token i is "open" (visible context)
          r_i ≈ 0   -> token i is "closed" (masked, to be predicted)

      * Masked tokens:

          c_i = r_i · z^{(S)}_i + (1 - r_i) · m

        where m ∈ R^D is a learned mask token shared across positions.

        Depending on where gating is applied:

          - If done "at the end" (gate_layer_index is None), c_i is built
            from the final encoder tokens and only affects the predictor input.

          - If done after an earlier transformer block, c_i replaces the
            tokens that feed into subsequent transformer blocks, so later
            layers operate on a masked sequence.

    Loss
    ----
    The prediction loss is an MSE over *all* tokens (gates only affect the
    predictor input / intermediate states, not which tokens are evaluated):

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

    Gating location
    ---------------
    Two knobs:

      * gate_layer_index ∈ {0, …, depth-1} or None

          - None: gating is applied on the final encoder tokens, *outside*
            the transformer. The encoder always sees the full unmasked image,
            only the predictor sees masked tokens.

          - k: we run the context encoder block-by-block. At block k we
            apply gating, and the remaining blocks see masked tokens.

      * gate_inside_block ∈ {False, True}

          - False (default):
               gate MLP is applied to the *output* of the entire block:
                   x ← Attn(x) + x
                   x ← FFN(x) + x
                   [gate here on x]

          - True:
               gate MLP is applied *inside* the block, after attention but
               before the residual connection and FFN:
                   y = Attn(x)
                   [gate here on y]
                   x ← y_gated + x
                   x ← FFN(x) + x

        This allows you to force the model to decide what to keep/drop earlier
        in the computation, before representations fully oversmooth.

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
        gate_layer_index: int | None = None,
        gate_location: str = "post",
    ) -> None:
        # Set up context/target encoders and EMA utilities
        super().__init__(encoder=encoder)

        self.patch_size = patch_size
        self.img_size = img_size
        self.lambda_gates = lambda_gates
        self.gate_exp_alpha = gate_exp_alpha

        # Where to apply gating inside the context encoder.
        # None => gating on final tokens (outside encoder).
        # k    => gating at transformer block k (location controlled by gate_location).
        self.gate_layer_index = gate_layer_index

        assert gate_location in ("attn", "skip", "post"), (
            f"gate_location must be one of 'attn', 'skip', 'post', "
            f"got {gate_location!r}"
        )
        self.gate_location = gate_location

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

        # Simple per-token gating head over embeddings:
        #   log α_{b,i} = wᵀ LN(z_{b,i}) + b
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(hidden_emb_dim),
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


    def _encode_student_with_internal_gating(
        self,
        img: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the context encoder block-by-block, applying gating at a given
        transformer block index (self.gate_layer_index). The remaining blocks
        see masked tokens.

        gate_location controls *where* in the block we gate:

          - "attn":
              y = Attn(x)
              [gate on y]
              x = y_gated + x
              x = FFN(x) + x

          - "skip":
              y = Attn(x)
              x_skip = y + x
              [gate on x_skip]
              x = x_gated
              x = FFN(x) + x

          - "post":
              x = Attn(x) + x
              x = FFN(x) + x
              [gate on x]

        Returns:
            student_tokens:          (B, N, D) final normalized tokens after all blocks
                                     (already influenced by gating).
            gate_values:             (B, N) sampled gate values in [0, 1]
            gate_probs:              (B, N) open probabilities π_{b,i}
            gate_mlp_input_example:  (N, D) the *token matrix* that was fed
                                     into gate_mlp, for a representative sample
                                     (used for SSIM observability).
        """
        vt = self.context_encoder
        device = img.device

        # Patch embedding + positional encoding (same as VisionTransformer.forward)
        x = vt.to_patch_embedding(img)  # (B, N, D)
        x = x + vt.pos_embedding.to(device, dtype=x.dtype)

        depth = len(vt.transformer.layers)
        gate_values: torch.Tensor | None = None
        gate_probs: torch.Tensor | None = None
        gate_mlp_input_example: torch.Tensor | None = None

        for layer_idx, (attn, ff) in enumerate(vt.transformer.layers):
            if layer_idx != self.gate_layer_index:
                # Standard transformer block
                x = attn(x) + x
                x = ff(x) + x
                continue

            # ---- Gated block at layer_idx == gate_layer_index ----
            if self.gate_location == "attn":
                # Gate on raw attention output before residual
                attn_out = attn(x)  # (B, N, D)
                gate_mlp_input = attn_out

                log_alpha = self.gate_mlp(gate_mlp_input).squeeze(-1)  # (B, N)
                log_alpha = torch.clamp(log_alpha, min=-10.0, max=10.0)

                gate_values, gate_probs = self.gate(
                    log_alpha,
                    training=self.training,
                )  # (B, N), (B, N)

                B, N, D = attn_out.shape
                mask_token = self.mask_token.to(device)          # (1, D)
                mask_tokens = mask_token.expand(B, N, D)         # (B, N, D)

                r = gate_values.unsqueeze(-1)                    # (B, N, 1)
                one_minus_r = 1.0 - r                            # (B, N, 1)

                attn_out_gated = r * attn_out + one_minus_r * mask_tokens
                x = attn_out_gated + x                           # residual
                x = ff(x) + x                                    # FFN + residual

            elif self.gate_location == "skip":
                # Gate after attention+skip, before FFN
                attn_out = attn(x)                    # (B, N, D)
                x_skip = attn_out + x                 # (B, N, D)
                gate_mlp_input = x_skip

                log_alpha = self.gate_mlp(gate_mlp_input).squeeze(-1)  # (B, N)
                log_alpha = torch.clamp(log_alpha, min=-10.0, max=10.0)

                gate_values, gate_probs = self.gate(
                    log_alpha,
                    training=self.training,
                )  # (B, N), (B, N)

                B, N, D = x_skip.shape
                mask_token = self.mask_token.to(device)          # (1, D)
                mask_tokens = mask_token.expand(B, N, D)         # (B, N, D)

                r = gate_values.unsqueeze(-1)                    # (B, N, 1)
                one_minus_r = 1.0 - r                            # (B, N, 1)

                x_gated = r * x_skip + one_minus_r * mask_tokens
                x = ff(x_gated) + x_gated                        # FFN + residual

            else:  # self.gate_location == "post"
                # Standard block, then gate on full block output
                x = attn(x) + x
                x = ff(x) + x
                gate_mlp_input = x

                log_alpha = self.gate_mlp(gate_mlp_input).squeeze(-1)  # (B, N)
                log_alpha = torch.clamp(log_alpha, min=-10.0, max=10.0)

                gate_values, gate_probs = self.gate(
                    log_alpha,
                    training=self.training,
                )  # (B, N), (B, N)

                B, N, D = x.shape
                mask_token = self.mask_token.to(device)          # (1, D)
                mask_tokens = mask_token.expand(B, N, D)         # (B, N, D)

                r = gate_values.unsqueeze(-1)                    # (B, N, 1)
                one_minus_r = 1.0 - r                            # (B, N, 1)

                x = r * x + one_minus_r * mask_tokens            # (B, N, D)

            # whatever gate_mlp_input we used, keep one example for observability
            gate_mlp_input_example = gate_mlp_input[0].detach()  # (N, D)

        # Final LayerNorm of the transformer
        x = vt.transformer.norm(x)                               # (B, N, D)

        if gate_values is None or gate_probs is None or gate_mlp_input_example is None:
            raise RuntimeError(
                "Internal gating requested but gate_layer_index was never hit."
            )

        return x, gate_values, gate_probs, gate_mlp_input_example


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
                * "gate_mlp_input_example"  # (N, D) tokens fed to gate_mlp
        """
        device = img.device

        # -------------------------
        # 1. Teacher / target path
        # -------------------------
        with torch.no_grad():
            target_features = self._encode_teacher(img)  # (B, N, D)

        B, N, D = target_features.shape

        # -------------------------
        # 2. Student path + gating location logic
        # -------------------------
        depth = len(self.context_encoder.transformer.layers)
        use_internal_gating = (
            self.gate_layer_index is not None
            and 0 <= self.gate_layer_index < depth
        )

        if use_internal_gating:
            # Gating happens inside the encoder after block gate_layer_index.
            # Remaining blocks see masked tokens. Predictor sees final masked tokens.
            (
                student_tokens,
                gate_values,
                gate_probs,
                gate_mlp_input_example,
            ) = self._encode_student_with_internal_gating(img)
            context_tokens = student_tokens  # already masked by encoder

        else:
            # Original behavior: context encoder sees full image;
            # gating only affects predictor input (post-encoder).
            student_tokens = self.context_encoder(img)          # (B, N, D)

            # Gate MLP input in this case is simply the final student tokens
            gate_mlp_input = student_tokens                    # (B, N, D)

            log_alpha = self.gate_mlp(gate_mlp_input).squeeze(-1)  # (B, N)
            log_alpha = torch.clamp(log_alpha, min=-10.0, max=10.0)

            gate_values, gate_probs = self.gate(
                log_alpha,
                training=self.training,
            )  # (B, N), (B, N)

            B, N, D = student_tokens.shape
            mask_token = self.mask_token.to(device)             # (1, D)
            mask_tokens = mask_token.expand(B, N, D)            # (B, N, D)

            r = gate_values.unsqueeze(-1)                       # (B, N, 1)
            one_minus_r = 1.0 - r                               # (B, N, 1)

            context_tokens = r * student_tokens + one_minus_r * mask_tokens

            # Example for SSIM observability
            gate_mlp_input_example = gate_mlp_input[0].detach()  # (N, D)

        # -------------------------
        # 3. Predictor
        # -------------------------
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
        # 4. Prediction loss on all tokens
        # -------------------------
        diff = predictor_output - target_features         # (B, N, D)
        sq_norm = (diff ** 2).sum(dim=-1)                 # (B, N)
        pred_loss = sq_norm.mean()                        # scalar

        # -------------------------
        # 5. Gate sparsity penalty (normalized + exponential)
        # -------------------------
        open_frac_per_sample = gate_probs.mean(dim=1)         # (B,)
        gate_penalty_per_sample = torch.exp(
            self.gate_exp_alpha * open_frac_per_sample
        ) - 1.0                                               # (B,)
        gate_penalty = gate_penalty_per_sample.mean()         # scalar

        total_loss = pred_loss + self.lambda_gates * gate_penalty

        # -------------------------
        # 6. Observability stats
        # -------------------------
        open_prob_mean = gate_probs.mean()                     # scalar
        closed_prob_mean = 1.0 - open_prob_mean
        closed_frac_per_sample = 1.0 - open_frac_per_sample    # (B,)

        H_p = self.num_patches_per_dim
        W_p = self.num_patches_per_dim
        if H_p * W_p != N:
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
            "gate_map_example": gate_map_example,
            "gate_values_full": gate_values.detach(),
            "gate_mlp_input_example": gate_mlp_input_example,  # (N, D)
        }

        return total_loss, stats

