"""
Tests for SinusSurgeryDataModule.

Run with:
    pytest tests/datamodule/test_datamodule.py -v

Integration tests — use real dataset. Skipped if data directory not found.
"""

import warnings
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from src.datamodule.datamodule import SinusSurgeryDataModule, VALID_EXPERIMENTS
from src.utils.logging import get_logger

log = get_logger(__name__)

DATA_ROOT = Path("data/uw-sinus-surgery-CL")
pytestmark = pytest.mark.skipif(
    not DATA_ROOT.exists(),
    reason=f"Dataset not found at {DATA_ROOT}",
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_dm(
    experiment: str, fold: int = 1, batch_size: int = 4, **kwargs
) -> SinusSurgeryDataModule:
    dm = SinusSurgeryDataModule(
        data_root=DATA_ROOT,
        experiment=experiment,
        fold=fold,
        batch_size=batch_size,
        num_workers=0,  # 0 workers — avoids multiprocessing overhead in tests
        **kwargs,
    )
    dm.setup()
    return dm


# ─────────────────────────────────────────────────────────────────────────────
# 1. Initialisation & validation
# ─────────────────────────────────────────────────────────────────────────────


class TestInit:

    def test_invalid_experiment_raises(self) -> None:
        with pytest.raises(AssertionError, match="experiment must be one of"):
            SinusSurgeryDataModule(DATA_ROOT, experiment="X->Y")

    def test_invalid_fold_raises_for_ll(self) -> None:
        with pytest.raises(AssertionError, match="fold must be 1, 2 or 3"):
            SinusSurgeryDataModule(DATA_ROOT, experiment="L->L", fold=5)

    def test_fold_ignored_with_warning_for_cc(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dm = SinusSurgeryDataModule(DATA_ROOT, experiment="C->C", fold=2)
        assert len(w) == 1
        assert "fold=2 has no effect" in str(w[0].message)
        assert dm.fold == 1

    @pytest.mark.parametrize("experiment", ["C->C", "C->L", "L->C"])
    def test_fold_default_no_warning(self, experiment: str) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SinusSurgeryDataModule(DATA_ROOT, experiment=experiment, fold=1)
        assert len(w) == 0

    def test_all_valid_experiments_initialise(self) -> None:
        for exp in VALID_EXPERIMENTS:
            fold = 1
            dm = SinusSurgeryDataModule(DATA_ROOT, experiment=exp, fold=fold)
            assert dm.experiment == exp


# ─────────────────────────────────────────────────────────────────────────────
# 2. Split counts — every experiment/fold has correct sizes
# ─────────────────────────────────────────────────────────────────────────────


class TestSplitCounts:

    def test_cc_counts(self) -> None:
        dm = make_dm("C->C")
        assert len(dm.train_ds) == 2908  # S01-S07
        assert len(dm.val_ds) == 291  # S08
        assert len(dm.test_ds) == 1146  # S09-S10

    @pytest.mark.parametrize(
        "fold,train,val,test",
        [
            (1, 2801, 703, 1154),  # train=L02, val=L03, test=L01
            (2, 1154, 703, 2801),  # train=L01, val=L03, test=L02
            (3, 1154, 2801, 703),  # train=L01, val=L02, test=L03
        ],
    )
    def test_ll_counts(self, fold: int, train: int, val: int, test: int) -> None:
        dm = make_dm("L->L", fold=fold)
        assert len(dm.train_ds) == train, f"fold={fold} train"
        assert len(dm.val_ds) == val, f"fold={fold} val"
        assert len(dm.test_ds) == test, f"fold={fold} test"

    def test_cl_counts(self) -> None:
        dm = make_dm("C->L")
        assert len(dm.train_ds) == 3735  # S01-S09
        assert len(dm.val_ds) == 610  # S10
        assert len(dm.test_ds) == 4658  # all live

    def test_lc_counts(self) -> None:
        dm = make_dm("L->C")
        assert len(dm.train_ds) == 3955  # L01+L02
        assert len(dm.val_ds) == 703  # L03
        assert len(dm.test_ds) == 4345  # all cadaver


# ─────────────────────────────────────────────────────────────────────────────
# 3. No data leakage — train/val/test stems are disjoint
# ─────────────────────────────────────────────────────────────────────────────


class TestNoLeakage:

    def _stems(self, ds) -> set:
        # Read stems directly from the cached path list — no image I/O needed.
        return {img_path.stem for img_path, _ in ds.samples}

    @pytest.mark.parametrize("experiment", ["C->C", "C->L", "L->C"])
    def test_train_test_disjoint(self, experiment: str) -> None:
        dm = make_dm(experiment)
        assert self._stems(dm.train_ds).isdisjoint(
            self._stems(dm.test_ds)
        ), f"{experiment}: train/test overlap"

    @pytest.mark.parametrize("experiment", ["C->C", "C->L", "L->C"])
    def test_val_test_disjoint(self, experiment: str) -> None:
        dm = make_dm(experiment)
        assert self._stems(dm.val_ds).isdisjoint(
            self._stems(dm.test_ds)
        ), f"{experiment}: val/test overlap"

    @pytest.mark.parametrize("experiment", ["C->C", "C->L", "L->C"])
    def test_train_val_disjoint(self, experiment: str) -> None:
        dm = make_dm(experiment)
        assert self._stems(dm.train_ds).isdisjoint(
            self._stems(dm.val_ds)
        ), f"{experiment}: train/val overlap"

    @pytest.mark.parametrize("fold", [1, 2, 3])
    def test_ll_all_splits_disjoint(self, fold: int) -> None:
        dm = make_dm("L->L", fold=fold)
        train = self._stems(dm.train_ds)
        val = self._stems(dm.val_ds)
        test = self._stems(dm.test_ds)
        assert train.isdisjoint(test), f"fold={fold}: train/test overlap"
        assert train.isdisjoint(val), f"fold={fold}: train/val overlap"
        assert val.isdisjoint(test), f"fold={fold}: val/test overlap"


# ─────────────────────────────────────────────────────────────────────────────
# 4. DataLoader output — batch shapes and dtypes
# ─────────────────────────────────────────────────────────────────────────────


class TestDataLoader:

    @pytest.fixture(scope="class")
    def cc_batch(self) -> dict:
        dm = make_dm("C->C", batch_size=4)
        return next(iter(dm.train_dataloader()))

    def test_dataloader_returns_dataloader(self) -> None:
        dm = make_dm("C->C")
        assert isinstance(dm.train_dataloader(), DataLoader)
        assert isinstance(dm.val_dataloader(), DataLoader)
        assert isinstance(dm.test_dataloader(), DataLoader)

    def test_train_shuffled(self) -> None:
        dm = make_dm("C->C")
        assert dm.train_dataloader().sampler.__class__.__name__ == "RandomSampler"

    def test_val_not_shuffled(self) -> None:
        dm = make_dm("C->C")
        assert dm.val_dataloader().sampler.__class__.__name__ == "SequentialSampler"

    def test_test_not_shuffled(self) -> None:
        dm = make_dm("C->C")
        assert dm.test_dataloader().sampler.__class__.__name__ == "SequentialSampler"

    def test_batch_image_shape(self, cc_batch: dict) -> None:
        assert cc_batch["image"].shape == (4, 3, 256, 256)

    def test_batch_mask_shape(self, cc_batch: dict) -> None:
        assert cc_batch["mask"].shape == (4, 1, 256, 256)

    def test_batch_mask_binary(self, cc_batch: dict) -> None:
        unique = cc_batch["mask"].unique().tolist()
        assert all(v in [0.0, 1.0] for v in unique), f"Non-binary: {unique}"

    def test_batch_box_prompt_shape(self, cc_batch: dict) -> None:
        assert cc_batch["box_prompt"].shape == (4, 4)

    def test_batch_point_prompt_shape(self, cc_batch: dict) -> None:
        assert cc_batch["point_prompt"].shape == (4, 2)

    def test_batch_image_dtype(self, cc_batch: dict) -> None:
        assert cc_batch["image"].dtype == torch.float32

    def test_batch_box_dtype(self, cc_batch: dict) -> None:
        assert cc_batch["box_prompt"].dtype == torch.int64

    def test_batch_text_prompt_in_batch(self, cc_batch: dict) -> None:
        assert "text_prompt" in cc_batch
        assert isinstance(cc_batch["text_prompt"], (list, tuple))
        assert all(isinstance(t, str) for t in cc_batch["text_prompt"])


# ─────────────────────────────────────────────────────────────────────────────
# 5. Augmentation — train has it, val/test do not
# ─────────────────────────────────────────────────────────────────────────────


class TestAugmentation:

    def test_train_dataset_has_augment(self) -> None:
        dm = make_dm("C->C")
        # Augmented pipeline has more transforms than non-augmented
        assert len(dm.train_ds.image_tf.transforms) > len(dm.val_ds.image_tf.transforms)

    def test_val_dataset_no_augment(self) -> None:
        dm = make_dm("C->C")
        assert len(dm.val_ds.image_tf.transforms) == len(dm.test_ds.image_tf.transforms)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Stage-aware setup — setup("fit") vs setup("test")
# ─────────────────────────────────────────────────────────────────────────────


class TestSetupStages:

    def test_fit_stage_creates_train_val_only(self) -> None:
        dm = SinusSurgeryDataModule(
            DATA_ROOT, experiment="C->C", batch_size=4, num_workers=0
        )
        dm.setup(stage="fit")
        assert dm.train_ds is not None
        assert dm.val_ds is not None
        assert dm.test_ds is None

    def test_test_stage_creates_test_only(self) -> None:
        dm = SinusSurgeryDataModule(
            DATA_ROOT, experiment="C->C", batch_size=4, num_workers=0
        )
        dm.setup(stage="test")
        assert dm.train_ds is None
        assert dm.val_ds is None
        assert dm.test_ds is not None

    def test_none_stage_creates_all(self) -> None:
        dm = SinusSurgeryDataModule(
            DATA_ROOT, experiment="C->C", batch_size=4, num_workers=0
        )
        dm.setup(stage=None)
        assert dm.train_ds is not None
        assert dm.val_ds is not None
        assert dm.test_ds is not None

    def test_prepare_data_passes_with_valid_root(self) -> None:
        dm = SinusSurgeryDataModule(
            DATA_ROOT, experiment="C->C", batch_size=4, num_workers=0
        )
        dm.prepare_data()  # must not raise

    def test_prepare_data_raises_on_missing_root(self) -> None:
        dm = SinusSurgeryDataModule(
            "data/nonexistent", experiment="C->C", batch_size=4, num_workers=0
        )
        with pytest.raises(AssertionError, match="data_root not found"):
            dm.prepare_data()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Full DataLoader iteration — every batch loads without error
# ─────────────────────────────────────────────────────────────────────────────


class TestFullIteration:
    """Iterates every batch of every split to catch corrupt files or shape errors."""

    @pytest.mark.parametrize(
        "split,loader_fn",
        [
            ("train", "train_dataloader"),
            ("val", "val_dataloader"),
            ("test", "test_dataloader"),
        ],
    )
    def test_full_iteration_cc(self, split: str, loader_fn: str) -> None:
        batch_size = 64
        dm = make_dm("C->C", batch_size=batch_size)
        loader: DataLoader = getattr(dm, loader_fn)()
        n_batches = len(loader)
        total_samples = 0

        log.info(f"C->C {split}: {n_batches} batches × up to {batch_size} samples")
        for i, batch in enumerate(loader):
            total_samples += batch["image"].shape[0]
            assert batch["image"].shape[1:] == (
                3,
                256,
                256,
            ), f"bad image shape at batch {i}"
            assert batch["mask"].shape[1:] == (
                1,
                256,
                256,
            ), f"bad mask shape at batch {i}"
            assert not torch.isnan(batch["image"]).any(), f"NaN in image at batch {i}"
            if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                log.info(f"  [{i + 1}/{n_batches}] samples so far: {total_samples}")

        log.info(f"C->C {split} done — {total_samples} total samples")
        ds = getattr(dm, f"{split}_ds")
        # drop_last only on train; val/test keep all samples
        if split == "train":
            assert total_samples == (len(ds) // batch_size) * batch_size
        else:
            assert total_samples == len(ds)
