"""
scripts/run_all_experiments.py — Run experiments defined in research_plan.yaml.

research_plan.yaml is the single source of truth for ALL configuration.
Edit the yaml; run this script. No other arguments needed.

Usage
-----
    uv run python scripts/run_all_experiments.py --plan research_plan.yaml
    uv run python scripts/run_all_experiments.py --plan research_plan.yaml --dry_run True
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import fire
import yaml

from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _load_plan(plan_path: Path) -> dict:
    with open(plan_path) as f:
        return yaml.safe_load(f)


def _load_config(plan: dict, plan_path: Path) -> dict[str, object]:
    """Read the ``config`` section from the plan and resolve special values."""
    cfg = plan.get("config")
    if not cfg:
        raise ValueError(f"'config' section missing from {plan_path}")
    cfg = dict(cfg)
    # null num_workers → use all available CPUs
    if cfg.get("num_workers") is None:
        cfg["num_workers"] = os.cpu_count()
    return cfg


def _load_experiments(plan: dict) -> list[dict[str, object]]:
    """Flatten hypotheses → experiments into an ordered list, attaching hypothesis_id.

    Merge order (later wins): global config < hypothesis config < experiment keys.
    Hypothesis-level ``config:`` block is merged first so experiment keys still win.
    """
    configs: list[dict[str, object]] = []
    for hyp in plan.get("hypotheses", []):
        hyp_cfg: dict[str, object] = hyp.get("config", {})   # per-hypothesis overrides
        for exp in hyp.get("experiments", []):
            entry = {**hyp_cfg, **exp}            # exp keys override hyp_cfg
            entry["hypothesis_id"] = hyp["id"]   # e.g. "H2"
            configs.append(entry)
    return configs


def _has_test_results(metrics_path: Path) -> bool:
    """Return True only if metrics.csv exists AND contains at least one test/dice row."""
    if not metrics_path.exists():
        return False
    with open(metrics_path) as f:
        return any("test/dice" in line for line in f)


def _build_cmd(exp_cfg: dict[str, object], global_cfg: dict[str, object]) -> list[str]:
    """
    Merge per-experiment keys (experiment, fold) with global config and
    build the ``python main.py`` subprocess command.
    """
    merged = {**global_cfg, **exp_cfg}   # exp keys override global if they clash
    tokens = [sys.executable, "main.py"]
    for k, v in merged.items():
        tokens += [f"--{k}", str(v)]
    return tokens


def main(plan: str, dry_run: bool = False) -> None:
    """
    Run all experiments defined in research_plan.yaml.

    Args:
        plan:    Path to research_plan.yaml (the single source of truth).
        dry_run: Print commands but do not execute.
    """
    setup_logging()

    plan_path = Path(plan)
    if not plan_path.exists():
        raise FileNotFoundError(f"research_plan.yaml not found: {plan_path}")

    plan_data = _load_plan(plan_path)
    global_cfg = _load_config(plan_data, plan_path)
    experiments = _load_experiments(plan_data)

    if not experiments:
        raise ValueError(f"No experiments found in {plan_path}")

    logger.info("=" * 60)
    logger.info("ESS Research — %d experiment(s) from %s", len(experiments), plan_path.name)
    logger.info("  loss=%s  epochs=%s  zero_shot=%s  wandb=%s",
                global_cfg.get("loss"), global_cfg.get("max_epochs"),
                global_cfg.get("zero_shot"), global_cfg.get("use_wandb"))
    logger.info("=" * 60)

    failed: list[str] = []

    for i, exp_cfg in enumerate(experiments, 1):
        hypothesis_id = exp_cfg["hypothesis_id"]       # e.g. "H2"
        experiment = exp_cfg["experiment"]
        fold = exp_cfg["fold"]
        run_name = f"{str(experiment).replace('->', '_')}_fold{fold}"
        run_label = f"{hypothesis_id} / {experiment} fold{fold}"

        # Output lives under save_dir/hypothesis_id/run_name/
        hyp_save_dir = str(Path(global_cfg["save_dir"]) / hypothesis_id)
        metrics_path = Path(hyp_save_dir) / run_name / "metrics.csv"

        if _has_test_results(metrics_path):
            logger.info("[%d/%d] SKIP — %s (results exist: %s)", i, len(experiments), run_label, metrics_path)
            continue

        # hypothesis_id is internal bookkeeping — strip it before calling main.py
        main_cfg = {k: v for k, v in exp_cfg.items() if k != "hypothesis_id"}
        cmd = _build_cmd(main_cfg, {**global_cfg, "save_dir": hyp_save_dir})
        logger.info("[%d/%d] %s", i, len(experiments), run_label)
        logger.info("  cmd: %s", " ".join(cmd))

        if dry_run:
            continue

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        result = subprocess.run(cmd, cwd=plan_path.parent, env=env)
        if result.returncode != 0:
            logger.error("[%d/%d] FAILED — %s (exit %d)", i, len(experiments), run_label, result.returncode)
            failed.append(run_label)
        else:
            logger.info("[%d/%d] OK — %s", i, len(experiments), run_label)

    logger.info("=" * 60)
    if dry_run:
        logger.info("Dry run complete.")
        return

    if failed:
        logger.error("Failed: %s", failed)
    else:
        logger.info("All %d experiment(s) completed.", len(experiments))

    subprocess.run([sys.executable, str(Path(__file__).parent / "research_report.py")])


if __name__ == "__main__":
    fire.Fire(main)
