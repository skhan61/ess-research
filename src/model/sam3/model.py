"""
sam3/model.py — SAM3 segmentation model.

Loads facebook/sam3 from HuggingFace and wraps it to satisfy the
BaseSegmentationModel contract:

    forward(batch) → FloatTensor (B, 1, H, W) logits

Two modes controlled by use_lora:
  use_lora=False (default): all weights frozen — zero-shot inference.
  use_lora=True:            base weights frozen, LoRA adapters trainable.

The dataset already normalises images with SAM3 constants so batch["image"]
is passed directly as pixel_values — no extra preprocessing needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Sam3Config, Sam3Model as HFSam3Model, Sam3Processor

from src.model.base import BaseSegmentationModel
from src.utils.logging import get_logger

logger = get_logger(__name__)


VALID_PROMPT_MODES = {"text", "box", "point", "all"}


class SAM3Model(BaseSegmentationModel):
    """
    SAM3 surgical instrument segmentation model.

    Loads ``facebook/sam3`` with all base weights frozen. Optionally injects
    LoRA adapters into the ViT attention projections for fine-tuning.

    Args:
        image_size:       Input spatial resolution (H = W).
                          Must be divisible by ViT patch_size (14).
                          Recommended: 336, 560, 1008.
        use_lora:         If True, inject LoRA adapters and make them trainable.
                          If False (default), all weights frozen — zero-shot mode.
        lora_rank:        LoRA rank r (ignored when use_lora=False).
        lora_alpha:       LoRA scaling alpha (ignored when use_lora=False).
        lora_dropout:     Dropout on LoRA paths (ignored when use_lora=False).
        prompt_mode:      Which prompt type(s) to pass to SAM3.
                          "text"  — text prompt only (default).
                          "box"   — bounding-box prompt only.
                          "point" — centre-of-mass point prompt only.
                          "all"   — text + box + point combined.
        pretrained_model: HuggingFace model ID.
    """

    def __init__(
        self,
        image_size: int = 336,
        use_lora: bool = False,
        lora_rank: int = 4,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        prompt_mode: str = "text",
        pretrained_model: str = "facebook/sam3",
    ) -> None:
        super().__init__()
        self.image_size = image_size
        assert prompt_mode in VALID_PROMPT_MODES, (
            f"prompt_mode must be one of {VALID_PROMPT_MODES}, got '{prompt_mode}'"
        )
        self.prompt_mode = prompt_mode

        assert image_size % 14 == 0, (
            f"image_size={image_size} must be divisible by ViT patch_size=14. "
            "Try 336, 560, or 1008."
        )

        logger.info("Loading SAM3 config from %s …", pretrained_model)
        config = Sam3Config.from_pretrained(pretrained_model)
        config.image_size = image_size

        logger.info("Loading SAM3 weights …")
        self.sam3: nn.Module = HFSam3Model.from_pretrained(
            pretrained_model,
            config=config,
            ignore_mismatched_sizes=True,
        )

        # Freeze all base weights (always — LoRA adds new trainable params on top)
        for param in self.sam3.parameters():
            param.requires_grad_(False)

        if use_lora:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.sam3 = get_peft_model(self.sam3, lora_cfg)
            trainable = sum(p.numel() for p in self.sam3.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.sam3.parameters())
            logger.info(
                "LoRA applied — trainable: %d / %d (%.2f%%)",
                trainable, total, 100.0 * trainable / total,
            )
        else:
            logger.info("SAM3 loaded — all parameters frozen (zero-shot mode).")

        # Processor is lazy-initialized on first forward() call.
        # Eager init here would create HF Rust tokenizer threads in the main
        # process before DataLoader forks workers, causing a fork deadlock.
        self._pretrained_model: str = pretrained_model
        self.__processor: Sam3Processor | None = None

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: collated batch dict from SinusSurgeryDataset.
                   Required keys: ``image`` (B, 3, H, W) float32,
                                  ``text_prompt``  list[str]         — always present.
                                  ``box_prompt``   LongTensor (B, 4) — used when prompt_mode in {"box", "all"}.
                                  ``point_prompt`` LongTensor (B, 2) — used when prompt_mode in {"point", "all"}.

        Returns:
            Mask logits FloatTensor (B, 1, H, W).
        """
        device: torch.device = batch["image"].device

        # Lazy-init processor on first forward() — after DataLoader workers are
        # already forked, so no Rust thread inheritance deadlock.
        if self.__processor is None:
            self.__processor = Sam3Processor.from_pretrained(self._pretrained_model)

        # SAM3 always requires text (input_ids). Spatial prompts (box/point) are
        # additive on top of text — the architecture has no text-free mode.
        use_box   = self.prompt_mode in ("box",  "all")
        use_point = self.prompt_mode in ("point", "all")

        B = batch["image"].shape[0]

        # Use the processor only for text tokenisation (input_ids / attention_mask).
        # Spatial prompts are NOT passed through the processor because it requires
        # actual image pixel data to normalise coordinates; instead we normalise
        # manually by dividing by image_size to get [0, 1] range, which is what
        # the SAM3 model expects.
        enc = self.__processor(
            text=list(batch["text_prompt"]),
            return_tensors="pt",
            padding=True,
        ).to(device)

        model_kwargs: dict = {
            "pixel_values": batch["image"],
            "input_ids": enc.input_ids,
            "attention_mask": enc.attention_mask,
        }
        if use_box:
            # (B, 1, 4) normalised to [0, 1]
            model_kwargs["input_boxes"] = (
                batch["box_prompt"].unsqueeze(1).float() / self.image_size
            )
        if use_point:
            # (B, 1, 1, 2) normalised to [0, 1]
            model_kwargs["input_points"] = (
                batch["point_prompt"].unsqueeze(1).unsqueeze(1).float() / self.image_size
            )
            model_kwargs["input_labels"] = torch.ones(
                B, 1, 1, dtype=torch.long, device=device,
            )

        outputs = self.sam3(**model_kwargs)

        # semantic_seg: (B, 1, H', W') — resize to dataset image_size if needed
        seg: torch.Tensor = outputs.semantic_seg
        if seg.shape[-2:] != (self.image_size, self.image_size):
            seg = F.interpolate(
                seg,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return seg  # (B, 1, H, W)
