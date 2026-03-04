"""
metrics/segmentation.py — Binary segmentation metrics via torchmetrics.

All metrics operate on logits (pre-sigmoid): the threshold of 0.5 is applied
to sigmoid(logits) internally.

Usage in LightningModule
------------------------
    # __init__
    self.val_metrics  = SegmentationMetrics(prefix="val/")
    self.test_metrics = SegmentationMetrics(prefix="test/")

    # *_step
    self.val_metrics.update(logits, mask.int())

    # on_*_epoch_end
    self.log_dict(self.val_metrics.compute(), prog_bar=True)
    self.val_metrics.reset()
"""

import torch
import torchmetrics
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
)


class SegmentationMetrics(torchmetrics.Metric):
    """
    Collection of binary segmentation metrics backed by torchmetrics.

    Metrics computed (all at threshold=0.5 on sigmoid(logits)):
        Dice  (= F1)  — primary segmentation quality metric
        IoU   (= Jaccard Index) — intersection over union
        Precision — fraction of predicted positives that are correct
        Recall    — fraction of actual positives detected

    Args:
        prefix:    String prepended to every metric key (e.g. ``"val/"``).
        threshold: Sigmoid threshold for binarising predictions (default 0.5).
    """

    full_state_update: bool = False

    def __init__(self, prefix: str = "", threshold: float = 0.5) -> None:
        super().__init__()
        self.prefix = prefix
        self._collection = MetricCollection(
            {
                "dice": BinaryF1Score(threshold=threshold),
                "iou": BinaryJaccardIndex(threshold=threshold),
                "precision": BinaryPrecision(threshold=threshold),
                "recall": BinaryRecall(threshold=threshold),
            },
            prefix=prefix,
        )

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """
        Accumulate predictions for epoch-level computation.

        Args:
            logits: (B, 1, H, W) raw model output (pre-sigmoid).
            target: (B, 1, H, W) binary mask as int tensor {0, 1}.
        """
        probs = logits.sigmoid()
        # torchmetrics handles arbitrary shapes — no manual flattening needed
        self._collection.update(probs, target)

    def compute(self) -> dict[str, torch.Tensor]:
        """
        Compute epoch-level metrics.

        Returns:
            Dict mapping metric name (with prefix) → scalar tensor.
        """
        return self._collection.compute()

    def reset(self) -> None:
        """Reset all internal accumulators for the next epoch."""
        self._collection.reset()
