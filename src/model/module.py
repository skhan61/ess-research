"""
module.py — LightningModule wrapper for any BaseSegmentationModel.

Handles training/val/test steps, pluggable loss, torchmetrics-based evaluation,
and prediction visualisation. The actual forward pass is fully delegated to the
wrapped model.
"""

from pathlib import Path

import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from src.losses.segmentation import CombinedLoss
from src.metrics.segmentation import SegmentationMetrics
from src.model.base import BaseSegmentationModel
from src.utils.logging import get_logger
from src.visualization.predictions import save_prediction_grid

logger = get_logger(__name__)


class SinusSurgeryModule(L.LightningModule):
    """
    LightningModule that wraps any BaseSegmentationModel.

    Pluggable components:
        loss_fn  — any nn.Module with signature forward(logits, target) → scalar.
                   Defaults to CombinedLoss (Dice + BCE via MONAI).
        metrics  — torchmetrics-based Dice, IoU, Precision, Recall.

    Args:
        model:        Instance of BaseSegmentationModel (e.g. SAM3Model).
        lr:           Learning rate for AdamW.
        loss_fn:      Loss function. Defaults to CombinedLoss().
        vis_dir:      Optional directory to save prediction grids after test epoch.
                      If None, visualisation is skipped.
        vis_samples:  Number of samples to include in each prediction grid.
    """

    def __init__(
        self,
        model: BaseSegmentationModel,
        lr: float = 1e-4,
        loss_fn: nn.Module | None = None,
        vis_dir: Path | None = None,
        vis_samples: int = 8,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.vis_dir = Path(vis_dir) if vis_dir is not None else None
        self.vis_samples = vis_samples

        self.save_hyperparameters(ignore=["model", "loss_fn"])

        # Loss — default to Dice+BCE combined (best for imbalanced medical seg)
        self._loss_fn: nn.Module = loss_fn if loss_fn is not None else CombinedLoss()

        # Separate metric instances per split to avoid cross-epoch state bleed
        self._val_metrics = SegmentationMetrics(prefix="val/")
        self._test_metrics = SegmentationMetrics(prefix="test/")

        # Buffers for visualisation (accumulated across test batches)
        self._vis_images: list[torch.Tensor] = []
        self._vis_masks: list[torch.Tensor] = []
        self._vis_logits: list[torch.Tensor] = []

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> torch.Tensor:
        """Delegate entirely to the wrapped model."""
        return self.model(batch)

    # ── Steps ─────────────────────────────────────────────────────────────────

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        if batch_idx == 0:
            logger.info(
                "train step 0 — image %s  mask %s  text=%r",
                tuple(batch["image"].shape),
                tuple(batch["mask"].shape),
                batch["text_prompt"][0],
            )
        B = batch["image"].shape[0]
        logits = self(batch)
        loss = self._loss_fn(logits, batch["mask"])
        self.log("train/loss", loss, prog_bar=True, batch_size=B)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        B = batch["image"].shape[0]
        logits = self(batch)
        loss = self._loss_fn(logits, batch["mask"])
        self.log("val/loss", loss, prog_bar=True, batch_size=B)
        self._val_metrics.update(logits, batch["mask"].int())

    def on_validation_epoch_end(self) -> None:
        results = self._val_metrics.compute()
        self.log_dict(results, prog_bar=True)
        self._val_metrics.reset()

    def on_test_start(self) -> None:
        logger.info("Test started — model on device: %s", next(self.model.parameters()).device)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        B = batch["image"].shape[0]
        logits = self(batch)
        loss = self._loss_fn(logits, batch["mask"])
        self.log("test/loss", loss, batch_size=B)
        self._test_metrics.update(logits, batch["mask"].int())

        # Accumulate first N samples for visualisation
        if self.vis_dir is not None:
            needed = self.vis_samples - sum(t.size(0) for t in self._vis_images)
            if needed > 0:
                self._vis_images.append(batch["image"][:needed].cpu())
                self._vis_masks.append(batch["mask"][:needed].cpu())
                self._vis_logits.append(logits[:needed].detach().cpu())

    def on_test_epoch_end(self) -> None:
        # ── Metrics ───────────────────────────────────────────────────────────
        results = self._test_metrics.compute()
        self.log_dict(results)
        self._test_metrics.reset()

        logger.info("Test metrics:")
        for k, v in results.items():
            logger.info("  %s = %.4f", k, v.item())

        # ── Visualisation ─────────────────────────────────────────────────────
        if self.vis_dir is not None and self._vis_images:
            save_path = self.vis_dir / "predictions.png"
            save_prediction_grid(
                images=torch.cat(self._vis_images),
                masks=torch.cat(self._vis_masks),
                logits=torch.cat(self._vis_logits),
                save_path=save_path,
                max_samples=self.vis_samples,
            )
            logger.info("Prediction grid saved → %s", save_path)

            # Log image to W&B if the logger is attached
            for lgr in self.loggers:
                if isinstance(lgr, WandbLogger):
                    lgr.log_image(
                        key="test/predictions",
                        images=[str(save_path)],
                        caption=["image | ground-truth | prediction"],
                    )

        # Clear buffers
        self._vis_images.clear()
        self._vis_masks.clear()
        self._vis_logits.clear()

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self) -> torch.optim.Optimizer:
        trainable = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable, lr=self.hparams.lr)
