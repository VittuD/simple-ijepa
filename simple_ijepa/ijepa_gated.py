# simple_ijepa/ijepa_gated.py

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from simple_ijepa.utils import trunc_normal_
from simple_ijepa.transformer import VisionTransformer
from simple_ijepa.gates import HardConcreteGate


class GatedIJEPA(nn.Module):
    r"""
    Gated variant of I-JEPA where the *student* decides which tokens to use
    as context and which to mask, via differentiable hard-concrete gates.

    High-level idea
    ---------------
    For an image x, we compute:

      * Teacher / target encoder (EMA of the student):

          z^{(T)} = h_\xi(x)  \in \mathbb{R}^{N \times D}

      * Student / context encoder:

          z^{(S)} = h_\theta(x)  \in \mathbb{R}^{N \times D}

      * Gating network on student tokens:

          \log \alpha = g_\phi(z^{(S)})  \in \mathbb{R}^{N}
          r \sim \mathrm{HardConcrete}(\log \alpha)  \in [0, 1]^N

        Each gate r_i is interpreted as:
          r_i \approx 1   -> token i is "open" (visible context)
          r_i \approx 0   -> token i is "closed" (to be predicted)

      * Masked input to the predictor:

          c_i = r_i \, z^{(S)}_i + (1 - r_i) \, m

        where m \in \mathbb{R}^D is a learned mask token shared across positions.
        Stacking all tokens:

          C = [c_1, \dots, c_N]^\top  \in \mathbb{R}^{N \times D}

      * Predictor:

          \hat{z} = p_\psi(C)  \in \mathbb{R}^{N \times D}

    Loss
    ----
    We define a prediction loss only over *closed* tokens (where 1 - r_i is
    large), and add a penalty that grows exponentially with the *fraction*
    of open gates.

    For a batch of size B, tokens i = 1..N:

      Prediction loss (per batch):

        L_{\mathrm{pred}} =
          \frac{1}{B} \sum_{b=1}^B
            \frac{
              \sum_{i=1}^N (1 - r_{b,i})
                \big\| \mathrm{LN}(\hat{z}_{b,i}) -
                        \mathrm{LN}(z^{(T)}_{b,i}) \big\|_2^2
            }{
              \sum_{i=1}^N (1 - r_{b,i}) + \varepsilon
            }

      Gate usage and exponential penalty:

        Let \pi_{b,i} = \mathbb{P}( r_{b,i} > 0 ) and define the expected
        fraction of open gates for sample b:

          f_b = \frac{1}{N} \sum_{i=1}^N \pi_{b,i}  \in [0,1].

        We then define an exponential gate penalty per sample:

          p_b = \exp(\alpha \, f_b) - 1,

        where \alpha > 0 controls how sharply the cost grows as f_b \to 1.

        The batch-averaged gate penalty is:

          L_{\mathrm{gates}} =
            \frac{1}{B} \sum_{b=1}^B p_b.

      Total loss:

        L_{\mathrm{total}} =
          L_{\mathrm{pred}} + \lambda_{\mathrm{gates}} \, L_{\mathrm{gates}}.

    Intuition:
      * L_pred encourages the model to accurately predict teacher embeddings
        for tokens it chooses to close (mask out from the predictor input).
      * L_gates discourages using too many open gates, with a *gentle*
        penalty for small fractions and a *sharply increasing* penalty as
        the open fraction approaches 1.

      This can be viewed as a rate–distortion tradeoff, or as a Lagrangian
      relaxation of a constrained problem:

        \min \mathbb{E}[L_{\mathrm{pred}}]
        \quad \text{s.t.} \quad
        \mathbb{E}[f_b] \leq \text{budget}.

    Observability
    -------------
    The forward pass additionally returns a dictionary of statistics:

      * "pred_loss":            L_pred (scalar)
      * "gate_penalty":         L_gates (scalar, before \lambda_{\mathrm{gates}})
      * "open_prob_mean":       \mathbb{E}_{b,i}[\pi_{b,i}]
      * "closed_prob_mean":     1 - open_prob_mean
      * "open_frac_per_sample": f_b  (shape: B)
      * "closed_frac_per_sample": 1 - f_b
      * "gate_map_example":     example gate map r_{b,i} reshaped to
                                (H_p, W_p) for qualitative inspection, where
                                H_p = W_p = image_size / patch_size.
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
    ):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.lambda_gates = lambda_gates
        self.gate_exp_alpha = gate_exp_alpha

        # Derived: number of patches per spatial dimension
        self.num_patches_per_dim = img_size // patch_size

        # Student / context encoder (trainable)
        self.context_encoder = encoder

        # Teacher / target encoder (EMA of student, no grad)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)

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

        # Simple per-token gating head over student embeddings:
        #   \log \alpha_{b,i} = w^\top \mathrm{LN}(z^{(S)}_{b,i}) + b
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(hidden_emb_dim),
            nn.Linear(hidden_emb_dim, 1),
        )

        # Hard-concrete gate that turns log_alpha into:
        #   r_{b,i} \in [0, 1] and
        #   \pi_{b,i} \approx \mathbb{P}(r_{b,i} > 0)
        self.gate = HardConcreteGate(
            beta=gate_beta,
            gamma=gate_gamma,
            zeta=gate_zeta,
        )

    @torch.inference_mode
    def _encode_teacher(self, img: torch.Tensor) -> torch.Tensor:
        """
        Encode the image with the teacher / target encoder.

        Returns:
            target_features: (B, N, D) LayerNorm-ed token embeddings.
        """
        target_features = self.target_encoder(img)
        target_features = F.layer_norm(
            target_features,
            (target_features.size(-1),),
        )
        return target_features

    def forward(
        self,
        img: torch.Tensor,
        *args,
        **kwargs,
    ):
        """
        Args:
            img: Input batch of images, shape (B, C, H, W).

        Returns:
            total_loss: scalar tensor, the gated I-JEPA loss:
                           L_total = L_pred + lambda_gates * L_gates

            stats: dict with monitoring information, containing keys:
                * "pred_loss"
                * "gate_penalty"
                * "open_prob_mean"
                * "closed_prob_mean"
                * "open_frac_per_sample"
                * "closed_frac_per_sample"
                * "gate_map_example"
        """
        device = img.device

        # -------------------------
        # 1. Teacher / target path
        # -------------------------
        with torch.no_grad():
            target_features = self._encode_teacher(img)  # (B, N, D)

        B, N, D = target_features.shape

        # -------------------------
        # 2. Student embeddings
        # -------------------------
        # context_encoder processes the *full* image, no manual masking here.
        student_tokens = self.context_encoder(img)  # (B, N, D)

        # -------------------------
        # 3. Gating over tokens
        # -------------------------
        # gate_mlp produces a scalar log_alpha per token:
        #   \log \alpha_{b,i} \in \mathbb{R}
        # shape: (B, N, 1) -> squeeze to (B, N)
        log_alpha = self.gate_mlp(student_tokens).squeeze(-1)  # (B, N)
        log_alpha = torch.clamp(log_alpha, min=-10.0, max=10.0)

        # r \in [0, 1] are the actual gate samples (differentiable);
        # p_open \in (0,1) are the probabilities \mathbb{P}(r > 0) used
        # for the gate usage penalty.
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

        r = gate_values.unsqueeze(-1)    # (B, N, 1)
        one_minus_r = 1.0 - r            # (B, N, 1)

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
        # Every token's prediction error contributes to the loss, so gates
        # cannot "opt out" of being predicted by opening.
        diff = predictor_output - target_features       # (B, N, D)
        sq_norm = (diff ** 2).sum(dim=-1)               # (B, N)

        # Simple mean over all tokens and batch
        pred_loss = sq_norm.mean()

        # -------------------------
        # 7. Gate sparsity penalty (normalized + exponential)
        # -------------------------
        # gate_probs_{b,i} ≈ P(r_{b,i} > 0)  => "open gate" probability.
        #
        # First compute per-sample *fraction* of open gates:
        #
        #   f_b = (1 / N) * sum_i gate_probs_{b,i}  in [0, 1]
        #
        # Then define an exponential penalty:
        #
        #   p_b = exp(alpha * f_b) - 1
        #
        # where alpha > 0 controls how sharply the cost grows as f_b → 1.
        # Finally:
        #
        #   L_gates = (1/B) * sum_b p_b
        #
        # and:
        #
        #   L_total = L_pred + lambda_gates * L_gates

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

        # Per-sample expected open fraction and closed fraction
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
            "gate_map_example": gate_map_example,     # (H_p, W_p)
            # NEW: full gate values for the whole batch (B, N)
            "gate_values_full": gate_values.detach(), # still on device; we'll .cpu() in trainer
        }

        return total_loss, stats


    # -------------------------
    # 9. EMA utilities
    # -------------------------
    def update_params(self, gamma: float) -> None:
        r"""
        EMA update of teacher encoder parameters and buffers:

            \xi \leftarrow \gamma \, \xi + (1 - \gamma) \, \theta

        where \theta are the parameters of the student (context_encoder)
        path. We only EMA the encoder weights here, in line with the
        original I-JEPA style.
        """
        with torch.no_grad():
            valid_types = [torch.float, torch.float16]

            # Parameters
            for o_param, t_param in self._get_params():
                if o_param.dtype in valid_types and t_param.dtype in valid_types:
                    t_param.data.lerp_(o_param.data, 1.0 - gamma)

            # Buffers (e.g. layer norm stats, if any)
            for o_buffer, t_buffer in self._get_buffers():
                if o_buffer.dtype in valid_types and t_buffer.dtype in valid_types:
                    t_buffer.data.lerp_(o_buffer.data, 1.0 - gamma)

    def copy_params(self) -> None:
        r"""
        Hard-copy student encoder parameters to teacher encoder:

            \xi \leftarrow \theta
        """
        for o_param, t_param in self._get_params():
            t_param.data.copy_(o_param)

        for o_buffer, t_buffer in self._get_buffers():
            t_buffer.data.copy_(o_buffer)

    def save_encoder(self, path: str) -> None:
        """
        Save the teacher / target encoder weights, analogous to the original
        IJEPA implementation.
        """
        torch.save(self.target_encoder.state_dict(), path)

    def _get_params(self):
        return zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        )

    def _get_buffers(self):
        return zip(
            self.context_encoder.buffers(),
            self.target_encoder.buffers(),
        )
