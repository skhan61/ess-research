"""
SinusSurgeryDataModule
----------------------
PyTorch Lightning DataModule for UW-Sinus-Surgery-C/L.

Supports all 4 experimental settings from the research plan:
    C→C  : In-domain cadaver
    L→L  : In-domain live (3-fold cross-validation)
    C→L  : Cross-domain — train cadaver, test live
    L→C  : Cross-domain — train live, test cadaver

Usage:
    dm = SinusSurgeryDataModule(
        data_root="data/uw-sinus-surgery-CL",
        experiment="L->L",
        fold=1,
        batch_size=8,
    )
    dm.setup()
    print(dm.train_dataloader())
"""

import warnings
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from src.datamodule.dataset import SinusSurgeryDataset


# ── Split definitions ─────────────────────────────────────────────────────────

# C→C: 80/20 by video. S08 held out from train as validation.
_CC = {
    "train": dict(
        split="cadaver", video_ids=["S01", "S02", "S03", "S04", "S05", "S06", "S07"]
    ),
    "val": dict(split="cadaver", video_ids=["S08"]),
    "test": dict(split="cadaver", video_ids=["S09", "S10"]),
}

# L→L: 3-fold CV by video.
# Each fold: 1 test video, 1 val video (from training), 1 train video.
_LL_FOLDS = {
    1: {
        "train": dict(split="live", video_ids=["L02"]),
        "val": dict(split="live", video_ids=["L03"]),
        "test": dict(split="live", video_ids=["L01"]),
    },
    2: {
        "train": dict(split="live", video_ids=["L01"]),
        "val": dict(split="live", video_ids=["L03"]),
        "test": dict(split="live", video_ids=["L02"]),
    },
    3: {
        "train": dict(split="live", video_ids=["L01"]),
        "val": dict(split="live", video_ids=["L02"]),
        "test": dict(split="live", video_ids=["L03"]),
    },
}

# C→L: Train on all cadaver (S10 held for val), test on all live.
_CL = {
    "train": dict(
        split="cadaver",
        video_ids=["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09"],
    ),
    "val": dict(split="cadaver", video_ids=["S10"]),
    "test": dict(split="live", video_ids=None),  # all live videos
}

# L→C: Train on L01+L02 (L03 held for val), test on all cadaver.
_LC = {
    "train": dict(split="live", video_ids=["L01", "L02"]),
    "val": dict(split="live", video_ids=["L03"]),
    "test": dict(split="cadaver", video_ids=None),  # all cadaver videos
}

VALID_EXPERIMENTS = {"C->C", "L->L", "C->L", "L->C"}


# ── DataModule ────────────────────────────────────────────────────────────────


class SinusSurgeryDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for all 4 experiments in the research plan.

    No inference (predict) is needed — every image in this dataset has a
    ground-truth label, so evaluation is always metric-based (Dice/IoU).

    Args:
        data_root   : str | Path
            Root of UW-Sinus-Surgery-C/L dataset.
            Expected layout:
                <data_root>/cadaver/images/*.jpg
                <data_root>/cadaver/labels/*.png
                <data_root>/live/images/*.jpg
                <data_root>/live/labels/*.png

        experiment  : "C->C" | "L->L" | "C->L" | "L->C"
            Which experimental setting to run.

        fold        : 1 | 2 | 3
            ONLY used when experiment="L->L". Ignored otherwise.

            The 3 live videos (L01, L02, L03) are rotated as follows:
                fold=1 → train=L02  val=L03  test=L01
                fold=2 → train=L01  val=L03  test=L02
                fold=3 → train=L01  val=L02  test=L03

            The test video is the held-out video for that fold.
            The val video is used only for early stopping — never for
            model selection or reporting final numbers.
            Final L→L result = mean ± std of Dice across all 3 folds.

            For C->C, C->L, L->C the splits are fixed and fold has no effect:
                C->C  → train=S01-S07  val=S08       test=S09-S10
                C->L  → train=S01-S09  val=S10        test=all live
                L->C  → train=L01-L02  val=L03        test=all cadaver

        batch_size  : int   Samples per batch.
        num_workers : int   DataLoader worker processes.
        image_size  : int   Resize both image and mask to this square size.
        text_prompt : str   Text prompt passed to every sample.
        pin_memory  : bool  Pin memory for faster GPU transfer.
    """

    def __init__(
        self,
        data_root: str | Path,
        experiment: str,
        fold: int,
        batch_size: int,
        num_workers: int,
        image_size: int,
        text_prompt: str,
        pin_memory: bool,
    ) -> None:
        super().__init__()
        assert experiment in VALID_EXPERIMENTS, (
            f"experiment must be one of {VALID_EXPERIMENTS}, got '{experiment}'"
        )
        if experiment == "L->L":
            assert fold in (1, 2, 3), (
                f"fold must be 1, 2 or 3 for L->L experiment, got {fold}"
            )
        else:
            if fold != 1:
                warnings.warn(
                    f"fold={fold} has no effect for experiment='{experiment}'. "
                    f"fold is only used for 'L->L'. Resetting to 1.",
                    UserWarning,
                    stacklevel=2,
                )
            fold = 1

        self.data_root = Path(data_root)
        self.experiment = experiment
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.text_prompt = text_prompt
        self.pin_memory = pin_memory

        # resolved at setup()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        # save hyperparameters for Lightning logging
        self.save_hyperparameters()

    def _resolve_config(self) -> dict[str, dict[str, str | list[str] | None]]:
        """Return the train/val/test video_ids config for this experiment/fold."""
        if self.experiment == "C->C":
            return _CC
        if self.experiment == "L->L":
            return _LL_FOLDS[self.fold]
        if self.experiment == "C->L":
            return _CL
        if self.experiment == "L->C":
            return _LC

    def prepare_data(self) -> None:
        """Verify that the data root exists. Called once before setup()."""
        assert self.data_root.exists(), (
            f"data_root not found: {self.data_root}. "
            "Download the UW-Sinus-Surgery-C/L dataset first."
        )
        assert (self.data_root / "cadaver").exists(), (
            f"Missing cadaver subdirectory: {self.data_root / 'cadaver'}"
        )
        assert (self.data_root / "live").exists(), (
            f"Missing live subdirectory: {self.data_root / 'live'}"
        )

    def setup(self, stage: str | None = None) -> None:
        cfg = self._resolve_config()

        if stage in ("fit", None):
            self.train_ds = SinusSurgeryDataset(
                data_root=self.data_root,
                augment=True,
                image_size=self.image_size,
                text_prompt=self.text_prompt,
                **cfg["train"],
            )
            self.val_ds = SinusSurgeryDataset(
                data_root=self.data_root,
                augment=False,
                image_size=self.image_size,
                text_prompt=self.text_prompt,
                **cfg["val"],
            )

        if stage in ("test", "predict", None):
            self.test_ds = SinusSurgeryDataset(
                data_root=self.data_root,
                augment=False,
                image_size=self.image_size,
                text_prompt=self.text_prompt,
                **cfg["test"],
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def __repr__(self) -> str:
        fold_str = f"  fold={self.fold}" if self.experiment == "L->L" else ""
        lines = [
            f"SinusSurgeryDataModule(",
            f"  experiment={self.experiment!r},{fold_str}",
            f"  batch_size={self.batch_size}, image_size={self.image_size}",
        ]
        if self.train_ds:
            lines += [
                f"  train → {self.train_ds}",
                f"  val   → {self.val_ds}",
                f"  test  → {self.test_ds}",
            ]
        lines.append(")")
        return "\n".join(lines)
