




"""
losses/bce.py — Binary Cross-Entropy loss (numerically stable).

Wraps nn.BCEWithLogitsLoss so it plugs into the BaseLoss contract.
"""

import torch
import torch.nn as nn

from src.losses.base import BaseLoss


class BCELoss(BaseLoss):
    """
    Binary Cross-Entropy loss applied to raw logits.

    Numerically stable: uses log-sum-exp trick internally via
    nn.BCEWithLogitsLoss.

    Args:
        pos_weight: Optional scalar weight for positive class pixels.
                    Useful when instrument pixels are rare (class imbalance).
    """

    def __init__(self, pos_weight: float | None = None) -> None:
        super().__init__()
        weight = torch.tensor([pos_weight]) if pos_weight is not None else None
        self._bce = nn.BCEWithLogitsLoss(pos_weight=weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) raw model output.
            target: (B, 1, H, W) binary mask in {0.0, 1.0}.

        Returns:
            Scalar BCE loss.
        """
        return self._bce(logits, target)
