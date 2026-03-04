"""
tests/model/test_sam3_model.py — Smoke tests for SAM3Model.

Marked @pytest.mark.slow because they require downloading facebook/sam3
(~3-5 GB) on first run.  Skip in fast CI with: pytest -m "not slow"
"""

from __future__ import annotations

import pytest
import torch

from src.model.sam3 import SAM3Model


@pytest.mark.slow
def test_forward_shape_and_dtype() -> None:
    """Forward pass returns (B, 1, H, W) float32 logits."""
    model = SAM3Model(image_size=336)
    model.eval()

    batch = {
        "image": torch.randn(2, 3, 336, 336),
        "text_prompt": ["surgical instrument", "surgical instrument"],
    }
    with torch.no_grad():
        out = model(batch)

    assert out.shape == (2, 1, 336, 336), f"unexpected shape {out.shape}"
    assert out.dtype == torch.float32, f"unexpected dtype {out.dtype}"


@pytest.mark.slow
def test_only_lora_params_require_grad() -> None:
    """All trainable parameters should be LoRA adapter weights."""
    model = SAM3Model(image_size=336)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) > 0, "No trainable parameters found"

    non_lora_trainable = [n for n in trainable if "lora_" not in n]
    assert non_lora_trainable == [], (
        f"Non-LoRA params are trainable: {non_lora_trainable[:5]}"
    )


@pytest.mark.slow
def test_invalid_image_size_raises() -> None:
    """image_size not divisible by 14 should raise AssertionError."""
    with pytest.raises(AssertionError, match="divisible by ViT patch_size"):
        SAM3Model(image_size=256)
