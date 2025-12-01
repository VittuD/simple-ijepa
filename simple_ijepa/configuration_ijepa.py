"""Model configuration dataclasses for I-JEPA variants."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IJEPAConfig:
    """
    Architecture + gating configuration shared by both baseline and gated
    I-JEPA models.

    All fields live under the `model` section of the Hydra config, so you
    can override them from the command line like:

        python -m scripts.train_ddp \\
            variant=gated \\
            model.hidden_dim=768 \\
            model.lambda_gates=1e-3 \\
            model.gate_layer_index=2 \\
            model.gate_location=attn
    """

    # Encoder / predictor architecture
    image_size: int = 96
    patch_size: int = 8
    hidden_dim: int = 512
    depth: int = 6
    heads: int = 6
    predictor_depth: int = 6
    predictor_heads: int = 6
    num_targets: int = 4

    # Gating-specific controls (used only when variant == "gated")
    lambda_gates: float = 1.0
    gate_exp_alpha: float = 4.0
    gate_layer_index: Optional[int] = None # Indexed from 0; None = last layer
    gate_location: str = "post" # attn, skip, post (see GatedIJEPA docstring) 

    # Hard-concrete gate hyperparameters (used by GatedIJEPA -> HardConcreteGate)
    gate_beta: float = 2.0 / 3.0
    gate_gamma: float = -0.1
    gate_zeta: float = 1.1
