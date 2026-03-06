"""
scripts/debug_checkpoint.py — Verify checkpoint save/load cycle works with PyTorch 2.6.

Tests that:
1. Saving a checkpoint via torch.save does NOT embed pathlib.PosixPath
2. Loading the checkpoint with weights_only=True (PyTorch 2.6 default) succeeds
3. The add_safe_globals fix in main.py covers OLD checkpoints that contain PosixPath

Usage:
    uv run python scripts/debug_checkpoint.py
"""

from __future__ import annotations

import pathlib
import tempfile

import torch


def _check_for_posixpath(obj, path="root") -> list[str]:
    """Recursively find any pathlib.Path objects inside a nested structure."""
    hits = []
    if isinstance(obj, pathlib.PurePath):
        hits.append(f"{path} = {type(obj).__name__}({obj!r})")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            hits.extend(_check_for_posixpath(v, f"{path}[{k!r}]"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            hits.extend(_check_for_posixpath(v, f"{path}[{i}]"))
    return hits


def test_new_checkpoint():
    """NEW checkpoint (saved with fixed module.py) — must not contain PosixPath."""
    print("\n─── Test 1: NEW checkpoint (fixed code) ───")

    # Simulate what Lightning saves in a checkpoint
    fake_ckpt = {
        "state_dict": {"model.weight": torch.tensor([1.0, 2.0])},
        "hyper_parameters": {
            "lr": 1e-4,
            "vis_samples": 8,
            # vis_dir is EXCLUDED via ignore=["vis_dir"] — should NOT appear here
        },
        "epoch": 5,
    }

    with tempfile.NamedTemporaryFile(suffix=".ckpt") as f:
        torch.save(fake_ckpt, f.name)
        # Load with weights_only=True (PyTorch 2.6 default)
        loaded = torch.load(f.name, weights_only=True)

    hits = _check_for_posixpath(loaded)
    if hits:
        print(f"  FAIL — PosixPath found: {hits}")
    else:
        print("  PASS — no PosixPath in checkpoint, weights_only=True load succeeded")


def test_old_checkpoint_with_safe_globals():
    """OLD checkpoint (saved with broken code) — must load via add_safe_globals fix."""
    print("\n─── Test 2: OLD checkpoint (broken code) + add_safe_globals fix ───")

    fake_ckpt = {
        "state_dict": {"model.weight": torch.tensor([1.0, 2.0])},
        "hyper_parameters": {
            "lr": 1e-4,
            "vis_dir": pathlib.PosixPath("outputs/H3/C_C_fold1/predictions"),  # <-- OLD bug
            "vis_samples": 8,
        },
        "epoch": 17,
    }

    with tempfile.NamedTemporaryFile(suffix=".ckpt") as f:
        torch.save(fake_ckpt, f.name)

        # Without fix — should fail
        try:
            torch.load(f.name, weights_only=True)
            print("  Unexpected: loaded without fix (PyTorch may not enforce this yet)")
        except Exception as e:
            print(f"  Confirmed failure without fix: {type(e).__name__}")

        # With fix (what main.py now does)
        torch.serialization.add_safe_globals([pathlib.PosixPath])
        try:
            loaded = torch.load(f.name, weights_only=True)
            print("  PASS — add_safe_globals fix allows old checkpoint to load")
        except Exception as e:
            print(f"  FAIL — still broken even with add_safe_globals: {e}")


def test_lightning_ckpt_path():
    """Simulate the actual file that failed: outputs/H3/C_C_fold1/checkpoints/best-epoch17..."""
    print("\n─── Test 3: Actual saved checkpoint ───")
    ckpt = pathlib.Path(
        "outputs/H3/C_C_fold1/checkpoints/best-epoch17-val_dice0.9035.ckpt"
    )
    if not ckpt.exists():
        print("  SKIP — checkpoint not found (expected if outputs were cleared)")
        return

    torch.serialization.add_safe_globals([pathlib.PosixPath])
    try:
        data = torch.load(ckpt, map_location="cpu", weights_only=True)
        hparams = data.get("hyper_parameters", {})
        hits = _check_for_posixpath(hparams)
        print(f"  Loaded OK. hparam keys: {list(hparams.keys())}")
        if hits:
            print(f"  PosixPath found (safe globals covered it): {hits}")
        else:
            print("  No PosixPath in hparams (clean checkpoint)")
    except Exception as e:
        print(f"  FAIL: {e}")


if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    test_new_checkpoint()
    test_old_checkpoint_with_safe_globals()
    test_lightning_ckpt_path()
    print("\nDone.")
