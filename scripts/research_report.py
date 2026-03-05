"""
scripts/research_report.py — Hypothesis-driven experiment report.

Reads research_plan.yaml (hypotheses + expected outcomes) and
outputs/summary.csv (actual results from every run), then prints:

    H1  Within-domain: Cadaver
    ─────────────────────────
    Hypothesis: SAM3 fine-tuned on cadaver achieves high Dice...
    Expected:   Dice > 0.70
    Results:
        C->C fold1   dice=0.7812  iou=0.6412  precision=0.8234  recall=0.7444
    Status: ● DONE

    H2  Within-domain: Live Surgery (3-fold CV)
    ───────────────────────────────────────────
        L->L fold1   dice=0.6123
        L->L fold2   (not yet run)
        L->L fold3   (not yet run)
        Average      dice=0.6123 ± 0.0000
    Status: ◑ PARTIAL (1/3 complete)

W&B connection
--------------
Run all experiments with W&B enabled so results appear in your dashboard too:

    uv run python scripts/run_all_experiments.py --use_wandb

Then open W&B → your project → create a Report → embed charts → share link.
This script gives the same view locally (works offline, no account needed).

Usage:
    uv run python scripts/research_report.py
    uv run python scripts/research_report.py --plan research_plan.yaml
"""

from __future__ import annotations

import csv
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import fire
import yaml

from src.utils.logging import setup_logging


def _load_plan(plan_path: Path) -> dict[str, Any]:
    """
    Load and parse research_plan.yaml.

    Args:
        plan_path: Path to YAML research plan file.

    Returns:
        Parsed plan dict.
    """
    with open(plan_path) as f:
        return yaml.safe_load(f)


def _load_summary(outputs_dir: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    """
    Load outputs/summary.csv and index rows by (experiment, fold).

    Args:
        outputs_dir: Root outputs directory.

    Returns:
        Dict mapping (experiment, fold) → metrics row.
    """
    summary_path = outputs_dir / "summary.csv"
    if not summary_path.exists():
        return {}
    index: dict[tuple[str, str, str, str], dict[str, str]] = {}
    with open(summary_path, newline="") as f:
        for row in csv.DictReader(f):
            index[(row["hypothesis"], row["experiment"], row["fold"], row.get("variant", ""))] = row
    return index


def _fmt_metrics(row: dict[str, str]) -> str:
    """
    Format a result row as a compact inline string.

    Args:
        row: Dict with keys dice, iou, precision, recall.

    Returns:
        Formatted string, e.g. ``dice=0.7812  iou=0.6412``.
    """
    return "  ".join(
        f"{k}={row[k]}"
        for k in ("dice", "iou", "precision", "recall")
        if row.get(k, "—") != "—"
    )


def main(
    plan: str = "research_plan.yaml",
    outputs_dir: str = "outputs",
) -> None:
    """
    Print a hypothesis-driven research report.

    Automatically regenerates outputs/summary.csv before printing.

    Args:
        plan:        Path to research_plan.yaml.
        outputs_dir: Root outputs directory.
    """
    setup_logging()

    plan_path = Path(plan)
    if not plan_path.exists():
        print(f"[error] {plan_path} not found.")
        return

    # Regenerate summary.csv from latest runs (silent)
    summary_script = Path(__file__).parent / "summarize_results.py"
    subprocess.run(
        [sys.executable, str(summary_script), f"--outputs_dir={outputs_dir}"],
        capture_output=True,
    )

    research = _load_plan(plan_path)
    results = _load_summary(Path(outputs_dir))

    W = 70

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    print("=" * W)
    print(f"  PROJECT  : {research.get('project', '')}")
    print(f"  DATASET  : {research.get('dataset', '')}")
    print(f"  METRIC   : {research.get('primary_metric', 'test/dice')}")
    print("=" * W)

    # ── Per-hypothesis sections ────────────────────────────────────────────────
    for hyp in research.get("hypotheses", []):
        hid        = hyp["id"]
        title      = hyp["title"]
        hypothesis = hyp.get("hypothesis", "").strip().replace("\n", " ")
        expected   = hyp.get("expected", "—")
        exps       = hyp.get("experiments", [])

        header = f"  {hid}  {title}"
        print()
        print(header)
        print("  " + "─" * (len(header) - 2))
        print(f"  Hypothesis : {hypothesis}")
        print(f"  Expected   : {expected}")
        print(f"  Results    :")

        dice_vals: list[float] = []
        complete = 0

        for cfg in exps:
            exp_name = str(cfg["experiment"])
            fold     = str(cfg["fold"])
            # Build variant suffix from any extra experiment keys (same logic as runner)
            own_keys = [k for k in cfg if k not in ("experiment", "fold")]
            extras   = {k: cfg[k] for k in own_keys}
            variant  = ("_" + "_".join(f"{k}{v}" for k, v in sorted(extras.items()))) if extras else ""
            row      = results.get((hid, exp_name, fold, variant))
            label    = f"{exp_name} fold{fold}" + (f" [{' '.join(f'{k}={v}' for k,v in sorted(extras.items()))}]" if extras else "")

            if row and row.get("dice", "—") != "—":
                print(f"      {label:<28}  {_fmt_metrics(row)}")
                try:
                    dice_vals.append(float(row["dice"]))
                except (ValueError, KeyError):
                    pass
                complete += 1
            else:
                print(f"      {label:<28}  (not yet run)")

        # Show average for multi-fold experiments (L→L)
        if len(exps) > 1 and dice_vals:
            avg = statistics.mean(dice_vals)
            std = statistics.stdev(dice_vals) if len(dice_vals) > 1 else 0.0
            print(f"      {'Average':<18}  dice={avg:.4f} ± {std:.4f}")

        # Status
        n = len(exps)
        if complete == 0:
            symbol, status = "○", "PENDING"
        elif complete < n:
            symbol, status = "◑", f"PARTIAL ({complete}/{n} complete)"
        else:
            symbol, status = "●", "DONE"

        print(f"  Status     : {symbol} {status}")

    # ── Footer ─────────────────────────────────────────────────────────────────
    print()
    print("=" * W)
    print("  Run experiments  : uv run python scripts/run_all_experiments.py")
    print("  Run with W&B     : uv run python scripts/run_all_experiments.py --use_wandb")
    print("  Update report    : uv run python scripts/research_report.py")
    print("=" * W)
    print()


if __name__ == "__main__":
    fire.Fire(main)
