"""
main.py — ESS Research entry point.

Lightning lifecycle:
    DataModule + LightningModule → Trainer.fit → Trainer.test

Usage:
    # Train from scratch (default: combo loss, C->C experiment)
    uv run python main.py --experiment "C->C" --max_epochs 50

    # Choose a different loss function
    uv run python main.py --loss dice --experiment "L->L" --fold 2

    # Test only from a saved checkpoint (reproduce results)
    uv run python main.py --experiment "C->C" --ckpt_path outputs/C_C_fold1/version_0/checkpoints/best.ckpt

    # L->L fold 2 with file logging
    uv run python main.py --experiment "L->L" --fold 2 --max_epochs 50 --log_file logs/ll_fold2.log
"""

import logging
import os
from pathlib import Path

# Prevent HuggingFace tokenizer Rust threads from deadlocking DataLoader workers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import fire
import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from src.datamodule.datamodule import SinusSurgeryDataModule
from src.losses.segmentation import get_loss
from src.model.module import SinusSurgeryModule
from src.model.sam3 import SAM3Model
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def main(
    # ── Experiment ────────────────────────────────────────────────────────────
    experiment: str,
    fold: int,
    # ── Data ──────────────────────────────────────────────────────────────────
    data_root: str,
    image_size: int,
    text_prompt: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    # ── Training ──────────────────────────────────────────────────────────────
    max_epochs: int,
    lr: float,
    loss: str,                            # "bce" | "dice" | "combo" | "focal"
    early_stopping_patience: int,
    # ── Output ────────────────────────────────────────────────────────────────
    save_dir: str,
    save_predictions: bool,
    vis_samples: int,
    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str,
    # ── W&B experiment tracking ───────────────────────────────────────────────
    use_wandb: bool,
    wandb_project: str,
    # ── LoRA fine-tuning (only active when zero_shot=False) ───────────────────
    use_lora: bool = False,
    lora_rank: int = 4,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.1,
    # ── Trainer precision ─────────────────────────────────────────────────────
    precision: str = "32-true",             # "16-mixed" halves VRAM for LoRA training
    # ── Checkpoint / reproduce (optional — None means not provided) ───────────
    zero_shot: bool = False,
    ckpt_path: str | None = None,
    log_file: str | None = None,
) -> None:
    """SAM3 + LoRA fine-tuning for surgical instrument segmentation."""

    setup_logging(
        level=getattr(logging, log_level.upper()),
        log_file=Path(log_file) if log_file else None,
    )

    run_name = f"{experiment.replace('->', '_')}_fold{fold}"
    logger.info("=" * 60)
    logger.info("ESS Research — %s", run_name)
    logger.info("Device: %s  |  GPU: %s",
                "cuda" if torch.cuda.is_available() else "cpu",
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "—")
    logger.info("Loss: %s", loss)
    logger.info("=" * 60)

    # ── 1. DataModule ─────────────────────────────────────────────────────────
    dm = SinusSurgeryDataModule(
        data_root=Path(data_root),
        experiment=experiment,
        fold=fold,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        text_prompt=text_prompt,
        pin_memory=pin_memory,
    )

    # ── 2. Loss ───────────────────────────────────────────────────────────────
    loss_fn = get_loss(loss)

    # ── 3. Loggers — CSVLogger always on; WandbLogger optional ───────────────
    # One flat directory per experiment — no version_X subdirectories.
    run_dir = Path(save_dir) / run_name
    ckpt_dir = run_dir / "checkpoints"
    vis_dir = run_dir / "predictions" if save_predictions else None
    logger.info("Saving to: %s", run_dir)

    csv_logger = CSVLogger(save_dir=str(run_dir), name="", version="")

    loggers: list = [csv_logger]
    if use_wandb:
        loggers.append(
            WandbLogger(
                project=wandb_project,
                name=run_name,
                group=experiment,
                tags=[experiment, f"fold{fold}", loss],
                save_dir=str(run_dir),
            )
        )
        logger.info("W&B enabled — project=%s  run=%s", wandb_project, run_name)

    # ── 4. Model ──────────────────────────────────────────────────────────────
    _model = SAM3Model(
        image_size=image_size,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    if ckpt_path:
        logger.info("Loading checkpoint: %s", ckpt_path)
        module = SinusSurgeryModule.load_from_checkpoint(
            ckpt_path,
            model=_model,
            loss_fn=loss_fn,
            vis_dir=vis_dir,
            vis_samples=vis_samples,
        )
    else:
        module = SinusSurgeryModule(
            model=_model,
            lr=lr,
            loss_fn=loss_fn,
            vis_dir=vis_dir,
            vis_samples=vis_samples,
        )

    # ── 5. Callbacks ──────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-epoch{epoch:02d}-val_dice{val/dice:.4f}",
            monitor="val/dice",
            mode="max",           # higher Dice = better
            save_last=True,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor="val/dice",
            patience=early_stopping_patience,
            mode="max",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ── 6. Trainer ────────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        precision=precision,
        num_sanity_val_steps=1,
        log_every_n_steps=10,
        logger=loggers,
        callbacks=callbacks,
        enable_progress_bar=True,
    )

    # ── 7. Fit + Test (or Zero-shot / Test-only from checkpoint) ─────────────
    if zero_shot:
        logger.info("Zero-shot mode — skipping training, evaluating pretrained SAM3")
        trainer.test(module, datamodule=dm)
    elif ckpt_path:
        logger.info("Test-only mode — skipping training")
        trainer.test(module, datamodule=dm)
    else:
        trainer.fit(module, datamodule=dm)
        # Use best checkpoint for final test — never last epoch weights
        trainer.test(module, datamodule=dm, ckpt_path="best")
        logger.info("Best checkpoint: %s", trainer.checkpoint_callback.best_model_path)


if __name__ == "__main__":
    fire.Fire(main)
