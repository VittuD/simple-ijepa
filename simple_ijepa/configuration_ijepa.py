"""Model configuration dataclasses for I-JEPA variants."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IJEPAConfig:
    """Architecture knobs for both baseline and gated I-JEPA."""

    # Encoder / predictor architecture
    image_size: int = 96
    patch_size: int = 8
    hidden_dim: int = 512
    depth: int = 6
    heads: int = 6
    predictor_depth: int = 6
    predictor_heads: int = 6
    num_targets: int = 4

    # Gating-specific controls
    lambda_gates: float = 1.0
    gate_exp_alpha: float = 4.0
    gate_layer_index: Optional[int] = None
    gate_location: str = "post"
