"""
Tests for SinusSurgeryDataset.

Run with:
    pytest src/data/test_dataset.py -v

These are integration tests — they use the real downloaded dataset.
Tests are skipped automatically if the data directory is not found.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.datamodule.dataset import (
    SinusSurgeryDataset,
    generate_box_prompt,
    generate_point_prompt,
)

# ── Data root — skip all tests if dataset not downloaded ──────────────────────
DATA_ROOT = Path("data/uw-sinus-surgery-CL")
pytestmark = pytest.mark.skipif(
    not DATA_ROOT.exists(), reason=f"Dataset not found at {DATA_ROOT}"
)

# ── Known ground-truth counts from our earlier inspection ─────────────────────
CADAVER_COUNTS = {
    "S01": 285,
    "S02": 532,
    "S03": 344,
    "S04": 162,
    "S05": 491,
    "S06": 688,
    "S07": 406,
    "S08": 291,
    "S09": 536,
    "S10": 610,
}
LIVE_COUNTS = {"L01": 1154, "L02": 2801, "L03": 703}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Discovery & counts
# ─────────────────────────────────────────────────────────────────────────────


class TestDiscovery:

    def test_cadaver_total_count(self):
        ds = SinusSurgeryDataset(DATA_ROOT, "cadaver")
        assert len(ds) == sum(
            CADAVER_COUNTS.values()
        ), f"Expected {sum(CADAVER_COUNTS.values())}, got {len(ds)}"

    def test_live_total_count(self):
        ds = SinusSurgeryDataset(DATA_ROOT, "live")
        assert len(ds) == sum(LIVE_COUNTS.values())

    @pytest.mark.parametrize("vid,expected", CADAVER_COUNTS.items())
    def test_cadaver_per_video_count(self, vid, expected):
        ds = SinusSurgeryDataset(DATA_ROOT, "cadaver", video_ids=[vid])
        assert len(ds) == expected, f"{vid}: expected {expected}, got {len(ds)}"

    @pytest.mark.parametrize("vid,expected", LIVE_COUNTS.items())
    def test_live_per_video_count(self, vid, expected):
        ds = SinusSurgeryDataset(DATA_ROOT, "live", video_ids=[vid])
        assert len(ds) == expected

    def test_video_ids_exposed(self):
        ds = SinusSurgeryDataset(DATA_ROOT, "cadaver", video_ids=["S01", "S03"])
        assert ds.video_ids == ["S01", "S03"]

    def test_invalid_split_raises(self):
        with pytest.raises(AssertionError):
            SinusSurgeryDataset(DATA_ROOT, "invalid")

    def test_empty_video_ids_raises(self):
        with pytest.raises(AssertionError):
            SinusSurgeryDataset(DATA_ROOT, "cadaver", video_ids=["S99"])


# ─────────────────────────────────────────────────────────────────────────────
# 2. Output shapes and dtypes
# ─────────────────────────────────────────────────────────────────────────────


class TestOutputContract:

    @pytest.fixture(scope="class")
    def sample_with_instrument(self):
        """First cadaver sample that contains an instrument."""
        ds = SinusSurgeryDataset(DATA_ROOT, "cadaver", video_ids=["S01"])
        for i in range(len(ds)):
            s = ds[i]
            if s["has_instrument"]:
                return s
        pytest.skip("No instrument sample found in S01")

    @pytest.fixture(scope="class")
    def sample_no_instrument(self):
        """First cadaver sample with no instrument (background-only frame)."""
        ds = SinusSurgeryDataset(DATA_ROOT, "cadaver", video_ids=["S01"])
        for i in range(len(ds)):
            s = ds[i]
            if not s["has_instrument"]:
                return s
        pytest.skip("No background-only sample found in S01")

    def test_image_shape(self, sample_with_instrument):
        img = sample_with_instrument["image"]
        assert img.shape == (3, 256, 256), f"Got {img.shape}"

    def test_image_dtype(self, sample_with_instrument):
        assert sample_with_instrument["image"].dtype == torch.float32

    def test_mask_shape(self, sample_with_instrument):
        assert sample_with_instrument["mask"].shape == (1, 256, 256)

    def test_mask_binary(self, sample_with_instrument):
        mask = sample_with_instrument["mask"]
        unique = mask.unique().tolist()
        assert all(v in [0.0, 1.0] for v in unique), f"Non-binary values: {unique}"

    def test_box_prompt_shape(self, sample_with_instrument):
        assert sample_with_instrument["box_prompt"].shape == (4,)

    def test_box_prompt_dtype(self, sample_with_instrument):
        assert sample_with_instrument["box_prompt"].dtype == torch.int64

    def test_point_prompt_shape(self, sample_with_instrument):
        assert sample_with_instrument["point_prompt"].shape == (2,)

    def test_text_prompt_type(self, sample_with_instrument):
        assert isinstance(sample_with_instrument["text_prompt"], str)
        assert len(sample_with_instrument["text_prompt"]) > 0

    def test_video_id_in_sample(self, sample_with_instrument):
        assert sample_with_instrument["video_id"] == "S01"

    def test_stem_in_sample(self, sample_with_instrument):
        stem = sample_with_instrument["stem"]
        assert stem.startswith("S01_"), f"Unexpected stem: {stem}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Prompt correctness
# ─────────────────────────────────────────────────────────────────────────────


class TestPrompts:

    def test_box_zeros_when_no_instrument(self):
        mask = np.zeros((240, 240), dtype=np.uint8)
        box = generate_box_prompt(mask)
        assert box.sum().item() == 0

    def test_point_zeros_when_no_instrument(self):
        mask = np.zeros((240, 240), dtype=np.uint8)
        pt = generate_point_prompt(mask)
        assert pt.sum().item() == 0

    def test_box_correct_coords(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:60] = 1  # instrument at rows 20-39, cols 30-59
        box = generate_box_prompt(mask)
        assert box.tolist() == [30, 20, 59, 39]  # [x1,y1,x2,y2]

    def test_point_inside_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:60] = 1
        pt = generate_point_prompt(mask)
        cx, cy = pt.tolist()
        assert mask[cy, cx] == 1, f"Point ({cx},{cy}) is outside the mask"

    def test_box_nonzero_with_instrument(self):
        ds = SinusSurgeryDataset(DATA_ROOT, "cadaver", video_ids=["S01"])
        for i in range(len(ds)):
            s = ds[i]
            if s["has_instrument"]:
                assert s["box_prompt"].sum().item() > 0
                break

    def test_box_zero_without_instrument(self):
        ds = SinusSurgeryDataset(DATA_ROOT, "cadaver", video_ids=["S01"])
        for i in range(len(ds)):
            s = ds[i]
            if not s["has_instrument"]:
                assert s["box_prompt"].sum().item() == 0
                break

    def test_box_within_image_bounds(self):
        ds = SinusSurgeryDataset(DATA_ROOT, "cadaver", video_ids=["S01"])
        for i in range(min(50, len(ds))):
            s = ds[i]
            if s["has_instrument"]:
                x1, y1, x2, y2 = s["box_prompt"].tolist()
                assert 0 <= x1 <= x2 < 256
                assert 0 <= y1 <= y2 < 256

    def test_text_prompt_default(self):
        ds = SinusSurgeryDataset(DATA_ROOT, "cadaver", video_ids=["S01"])
        assert ds[0]["text_prompt"] == "surgical instrument"

    def test_text_prompt_custom(self):
        ds = SinusSurgeryDataset(
            DATA_ROOT, "cadaver", video_ids=["S01"], text_prompt="endoscopic tool"
        )
        assert ds[0]["text_prompt"] == "endoscopic tool"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Experiment splits — no data leakage
# ─────────────────────────────────────────────────────────────────────────────


class TestSplits:

    # Cadaver C→C
    def test_cadaver_train_test_no_overlap(self):
        train = SinusSurgeryDataset(
            DATA_ROOT,
            "cadaver",
            video_ids=["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08"],
        )
        test = SinusSurgeryDataset(DATA_ROOT, "cadaver", video_ids=["S09", "S10"])
        train_stems = {ds["stem"] for ds in train}
        test_stems = {ds["stem"] for ds in test}
        assert train_stems.isdisjoint(test_stems), "Train/test overlap detected!"

    def test_cadaver_train_test_covers_all(self):
        train = SinusSurgeryDataset(
            DATA_ROOT,
            "cadaver",
            video_ids=["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08"],
        )
        test = SinusSurgeryDataset(DATA_ROOT, "cadaver", video_ids=["S09", "S10"])
        full = SinusSurgeryDataset(DATA_ROOT, "cadaver")
        assert len(train) + len(test) == len(full)

    # Live L→L 3-fold CV
    FOLDS = {
        1: {"train": ["L02", "L03"], "test": ["L01"]},
        2: {"train": ["L01", "L03"], "test": ["L02"]},
        3: {"train": ["L01", "L02"], "test": ["L03"]},
    }

    @pytest.mark.parametrize("fold", [1, 2, 3])
    def test_live_fold_no_overlap(self, fold):
        cfg = self.FOLDS[fold]
        train = SinusSurgeryDataset(DATA_ROOT, "live", video_ids=cfg["train"])
        test = SinusSurgeryDataset(DATA_ROOT, "live", video_ids=cfg["test"])
        train_stems = {ds["stem"] for ds in train}
        test_stems = {ds["stem"] for ds in test}
        assert train_stems.isdisjoint(test_stems), f"Fold {fold}: overlap!"

    @pytest.mark.parametrize("fold", [1, 2, 3])
    def test_live_fold_covers_all(self, fold):
        cfg = self.FOLDS[fold]
        train = SinusSurgeryDataset(DATA_ROOT, "live", video_ids=cfg["train"])
        test = SinusSurgeryDataset(DATA_ROOT, "live", video_ids=cfg["test"])
        full = SinusSurgeryDataset(DATA_ROOT, "live")
        assert len(train) + len(test) == len(full)

    # Cross-domain C→L
    def test_cross_domain_CL_no_overlap(self):
        train = SinusSurgeryDataset(DATA_ROOT, "cadaver")
        test = SinusSurgeryDataset(DATA_ROOT, "live")
        train_stems = {ds["stem"] for ds in train}
        test_stems = {ds["stem"] for ds in test}
        assert train_stems.isdisjoint(test_stems)
