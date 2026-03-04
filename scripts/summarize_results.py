"""
scripts/summarize_results.py — Aggregate test metrics across all experiments.

Scans outputs/ for every metrics.csv produced by CSVLogger and extracts the
final test-epoch row (Dice, IoU, Precision, Recall, Loss) for each run.
Writes a summary table to outputs/summary.csv and prints it to stdout.

Usage:
    uv run python scripts/summarize_results.py
    uv run python scripts/summarize_results.py --outputs_dir my_outputs
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import fire


# Columns we care about (CSVLogger logs them with these exact keys)
_TEST_COLS = [
    "test/dice",
    "test/iou",
    "test/precision",
    "test/recall",
    "test/loss",
]


def _parse_run_meta(metrics_csv: Path) -> dict[str, str]:
    """
    Extract experiment name, fold, and hypothesis id from the path structure.

    Expected path pattern (current):
        outputs/<hypothesis_id>/<experiment>_fold<n>/metrics.csv

    Args:
        metrics_csv: Path to a metrics.csv file.

    Returns:
        Dict with keys: experiment, fold, hypothesis.
    """
    # e.g. C_C_fold1, L_L_fold2, C_L_fold1, L_C_fold1
    run_dir    = metrics_csv.parent.name         # e.g. "C_C_fold1"
    hypothesis = metrics_csv.parent.parent.name  # e.g. "H1"

    match = re.match(r"(.+)_fold(\d+)$", run_dir)
    if match:
        experiment = match.group(1).replace("_", "->", 1)  # C_C → C->C
        fold = match.group(2)
    else:
        experiment = run_dir
        fold = "?"

    return {"experiment": experiment, "fold": fold, "hypothesis": hypothesis}


def _extract_test_row(metrics_csv: Path) -> dict[str, str] | None:
    """
    Extract the last row that has any test/* metric populated.

    CSVLogger writes one row per step/epoch; test metrics only appear in
    the final test rows. We take the last non-empty test row.

    Args:
        metrics_csv: Path to a CSVLogger metrics.csv file.

    Returns:
        Dict mapping column name → value string, or None if no test data.
    """
    best_row: dict[str, str] | None = None

    with open(metrics_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if any(row.get(col, "").strip() for col in _TEST_COLS):
                best_row = row

    return best_row


def main(outputs_dir: str = "outputs") -> None:
    """
    Scan outputs_dir and print a summary table of test metrics.

    Args:
        outputs_dir: Root directory containing experiment subdirectories.
    """
    root = Path(outputs_dir)
    if not root.exists():
        print(f"[warn] {root} does not exist — no results to summarise.")
        return

    csv_files = sorted(root.glob("*/*/metrics.csv"))
    if not csv_files:
        print(f"[warn] No metrics.csv files found under {root}.")
        return

    rows: list[dict[str, str]] = []
    for csv_path in csv_files:
        meta = _parse_run_meta(csv_path)
        test_row = _extract_test_row(csv_path)
        if test_row is None:
            continue  # run has no test results yet

        record: dict[str, str] = {**meta}
        for col in _TEST_COLS:
            val = test_row.get(col, "").strip()
            record[col.replace("test/", "")] = f"{float(val):.4f}" if val else "—"
        rows.append(record)

    if not rows:
        print("No completed test runs found.")
        return

    # ── Print table ───────────────────────────────────────────────────────────
    headers = ["hypothesis", "experiment", "fold", "dice", "iou", "precision", "recall", "loss"]
    col_w = {h: max(len(h), max(len(r.get(h, "—")) for r in rows)) for h in headers}

    sep = "+" + "+".join("-" * (col_w[h] + 2) for h in headers) + "+"
    header_row = "|" + "|".join(f" {h:<{col_w[h]}} " for h in headers) + "|"

    print(sep)
    print(header_row)
    print(sep)
    for r in rows:
        print("|" + "|".join(f" {r.get(h, '—'):<{col_w[h]}} " for h in headers) + "|")
    print(sep)

    # ── Write summary CSV ─────────────────────────────────────────────────────
    summary_path = root / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    fire.Fire(main)
