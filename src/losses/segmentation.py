"""
losses/segmentation.py — Pluggable loss functions for binary segmentation.

All losses follow the same contract:
    forward(logits, target) → scalar

Where:
    logits : FloatTensor (B, 1, H, W) — raw model output (pre-sigmoid)
    target : FloatTensor (B, 1, H, W) — binary GT mask in {0.0, 1.0}

All implementations use MONAI — tested, production-grade, medically validated.

Factory
-------
    loss_fn = get_loss("combo")  # → CombinedLoss
    loss_fn = get_loss("dice")   # → DiceLoss
    loss_fn = get_loss("bce")    # → BCELoss
    loss_fn = get_loss("focal")  # → FocalLoss
"""

import torch
import torch.nn as nn
from monai.losses import DiceCELoss as _MonaiDiceCELoss
from monai.losses import DiceLoss as _MonaiDiceLoss
from monai.losses import FocalLoss as _MonaiFocalLoss

# ── Re-export MONAI losses with cleaner names ──────────────────────────────────

# Binary BCE loss (numerically stable via logits)
BCELoss = nn.BCEWithLogitsLoss


class _DiceLossWrapper(nn.Module):
    """
    MONAI DiceLoss wrapper that matches our (logits, target) contract.

    MONAI DiceLoss expects probabilities by default; we apply sigmoid internally.

    Args:
        smooth_nr: Numerator smoothing constant (default 1e-5).
        smooth_dr: Denominator smoothing constant (default 1e-5).
    """

    def __init__(self, smooth_nr: float = 1e-5, smooth_dr: float = 1e-5) -> None:
        super().__init__()
        self._fn = _MonaiDiceLoss(sigmoid=True, smooth_nr=smooth_nr, smooth_dr=smooth_dr)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) raw model output.
            target: (B, 1, H, W) binary mask in {0.0, 1.0}.

        Returns:
            Scalar Dice loss in [0, 1].
        """
        return self._fn(logits, target)


class _CombinedLossWrapper(nn.Module):
    """
    MONAI DiceCELoss: Dice + Binary Cross-Entropy (equal weight by default).

    Best default for imbalanced binary segmentation (surgical instruments ~15%
    of pixels). BCE handles calibration; Dice handles overlap quality.

    Args:
        lambda_dice: Weight for Dice term (default 1.0).
        lambda_ce:   Weight for BCE term (default 1.0).
    """

    def __init__(self, lambda_dice: float = 1.0, lambda_ce: float = 1.0) -> None:
        super().__init__()
        self._fn = _MonaiDiceCELoss(
            sigmoid=True,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) raw model output.
            target: (B, 1, H, W) binary mask in {0.0, 1.0}.

        Returns:
            Scalar combined loss.
        """
        return self._fn(logits, target)


class _FocalLossWrapper(nn.Module):
    """
    MONAI FocalLoss for highly imbalanced classes.

    Focuses training on hard misclassified pixels by down-weighting easy ones.

    Args:
        gamma: Focusing parameter (default 2.0). Higher → more focus on hard pixels.
    """

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self._fn = _MonaiFocalLoss(use_softmax=False, gamma=gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) raw model output.
            target: (B, 1, H, W) binary mask in {0.0, 1.0}.

        Returns:
            Scalar focal loss.
        """
        return self._fn(logits.sigmoid(), target)


# ── Canonical names ────────────────────────────────────────────────────────────

DiceLoss = _DiceLossWrapper
CombinedLoss = _CombinedLossWrapper
FocalLoss = _FocalLossWrapper


# ── Factory function ───────────────────────────────────────────────────────────

_LOSS_REGISTRY: dict[str, type[nn.Module]] = {
    "bce": BCELoss,
    "dice": DiceLoss,
    "combo": CombinedLoss,
    "focal": FocalLoss,
}


def get_loss(name: str, **kwargs: object) -> nn.Module:
    """
    Instantiate a loss function by name.

    Args:
        name:   One of ``"bce"``, ``"dice"``, ``"combo"``, ``"focal"``.
        **kwargs: Forwarded to the loss constructor.

    Returns:
        Instantiated ``nn.Module`` loss function.

    Raises:
        ValueError: If ``name`` is not a registered loss.
    """
    if name not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss {name!r}. Available: {list(_LOSS_REGISTRY)}"
        )
    return _LOSS_REGISTRY[name](**kwargs)
