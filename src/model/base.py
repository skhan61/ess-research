"""
base.py — Abstract base class for all segmentation models.

Every model (SAM3, etc.) must subclass BaseSegmentationModel
and implement forward().
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseSegmentationModel(ABC, nn.Module):
    """
    Contract every segmentation model must satisfy.

    Input  : batch dict with keys — image, box_prompt, point_prompt, text_prompt
    Output : predicted mask FloatTensor (B, 1, H, W)  logits, NOT sigmoid
    """

    @abstractmethod
    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict from SinusSurgeryDataset.__getitem__ collated into a batch.
                   Keys: image (B,3,H,W), mask (B,1,H,W), box_prompt (B,4),
                         point_prompt (B,2), text_prompt list[str]

        Returns:
            Predicted mask logits — FloatTensor (B, 1, H, W).
            Apply sigmoid for probabilities; use with BCEWithLogitsLoss.
        """
        ...
