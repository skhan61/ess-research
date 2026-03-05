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
import shutil
import subprocess
import sys
from pathlib import Path

import fire
import yaml

from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# ANSI helpers — only emit color when stdout is a TTY
import sys as _sys
_TTY   = _sys.stdout.isatty()
_R     = "\033[0m"   if _TTY else ""
_BOLD  = "\033[1m"   if _TTY else ""
_GREEN = "\033[32m"  if _TTY else ""
_CYAN  = "\033[36m"  if _TTY else ""
_YELL  = "\033[33m"  if _TTY else ""
_RED   = "\033[31m"  if _TTY else ""


def _banner(text: str, color: str = _CYAN) -> None:
    """Print a visually distinct section header."""
    line = "─" * 60
    logger.info("%s%s%s%s%s", color, _BOLD, line, _R, "")
    logger.info("%s%s  %s%s", color, _BOLD, text, _R)
    logger.info("%s%s%s%s%s", color, _BOLD, line, _R, "")


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
            # Keys defined directly on this experiment entry (beyond experiment/fold)
            entry["_exp_own_keys"] = [k for k in exp if k not in ("experiment", "fold")]
            configs.append(entry)
    return configs


def _has_test_results(metrics_path: Path) -> bool:
    """Return True only if metrics.csv exists AND contains at least one test/dice row."""
    if not metrics_path.exists():
        return False
    with open(metrics_path) as f:
        return any("test/dice" in line for line in f)


def _find_best_checkpoint(ckpt_dir: Path) -> Path | None:
    """Return the best checkpoint (highest val_dice in filename), or None."""
    if not ckpt_dir.exists():
        return None
    ckpts = [c for c in ckpt_dir.glob("best-epoch*.ckpt")]
    if not ckpts:
        return None
    # Sort by val_dice value embedded in filename: best-epochXX-val_diceY.ckpt
    def _dice(p: Path) -> float:
        try:
            return float(p.stem.split("val_dice")[-1])
        except ValueError:
            return 0.0
    return max(ckpts, key=_dice)


def _build_cmd(exp_cfg: dict[str, object], global_cfg: dict[str, object],
               ckpt_path: Path | None = None) -> list[str]:
    """
    Merge per-experiment keys (experiment, fold) with global config and
    build the ``python main.py`` subprocess command.
    """
    merged = {**global_cfg, **exp_cfg}   # exp keys override global if they clash
    tokens = [sys.executable, "main.py"]
    for k, v in merged.items():
        tokens += [f"--{k}", str(v)]
    if ckpt_path is not None:
        tokens += ["--ckpt_path", str(ckpt_path)]
    return tokens


def main(plan: str, dry_run: bool = False, test_only: bool = False,
         debug: bool = False) -> None:
    """
    Run all experiments defined in research_plan.yaml.

    Args:
        plan:      Path to research_plan.yaml (the single source of truth).
        dry_run:   Print commands but do not execute.
        test_only: Skip training — use existing best checkpoint for test phase only.
                   Runs that have no checkpoint are skipped with a warning.
        debug:     Run full train→val→test on 2 batches per split, 1 epoch.
                   All experiments run regardless of existing results.
                   Saves to outputs/debug/ — never touches real outputs.
    """
    # Clear stale debug outputs before setting up logging so the log file
    # is never deleted after it is opened.
    if debug:
        debug_root = Path("outputs/debug")
        if debug_root.exists():
            shutil.rmtree(debug_root)
        debug_root.mkdir(parents=True, exist_ok=True)
        setup_logging(log_file=debug_root / "runner.log")
    else:
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
        own_keys = exp_cfg.get("_exp_own_keys", [])
        extras = {k: exp_cfg[k] for k in own_keys}
        run_name = f"{str(experiment).replace('->', '_')}_fold{fold}"
        if extras:
            run_name += "_" + "_".join(f"{k}{v}" for k, v in sorted(extras.items()))
        run_label = f"{hypothesis_id} / {experiment} fold{fold}" + (
            "  [" + "  ".join(f"{k}={v}" for k, v in sorted(extras.items())) + "]"
            if extras else ""
        )

        # Output lives under save_dir/hypothesis_id/run_name/
        hyp_save_dir = str(Path(global_cfg["save_dir"]) / hypothesis_id)
        metrics_path = Path(hyp_save_dir) / run_name / "metrics.csv"

        if not debug and _has_test_results(metrics_path):
            logger.info("[%d/%d] SKIP — %s (results exist: %s)", i, len(experiments), run_label, metrics_path)
            continue

        # Strip internal bookkeeping keys before calling main.py
        main_cfg = {k: v for k, v in exp_cfg.items() if k not in ("hypothesis_id", "_exp_own_keys")}

        if debug:
            hyp_save_dir = str(Path("outputs/debug") / hypothesis_id)
            merged_cfg = {**global_cfg, "save_dir": hyp_save_dir,
                          "max_epochs": 1,
                          "limit_train_batches": 2,
                          "limit_val_batches": 2,
                          "limit_test_batches": 2}
        else:
            merged_cfg = {**global_cfg, "save_dir": hyp_save_dir}

        ckpt_path: Path | None = None
        if test_only:
            ckpt_dir = Path(hyp_save_dir) / run_name / "checkpoints"
            ckpt_path = _find_best_checkpoint(ckpt_dir)
            if ckpt_path is None:
                logger.warning("[%d/%d] SKIP (test_only, no checkpoint) — %s", i, len(experiments), run_label)
                continue
            _banner(f"[{i}/{len(experiments)}] TEST-ONLY  {run_label}  ckpt={ckpt_path.name}", _CYAN)
        else:
            _banner(f"[{i}/{len(experiments)}] START  {run_label}", _CYAN)

        cmd = _build_cmd(main_cfg, merged_cfg, ckpt_path=ckpt_path)
        logger.info("  cmd: %s", " ".join(cmd))

        if dry_run:
            continue

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"

        if debug:
            # In debug mode, tee subprocess output to a per-experiment log file
            # so errors are captured even when the process crashes.
            exp_log = Path("outputs/debug") / f"{run_name}.log"
            with open(exp_log, "w") as lf:
                proc = subprocess.Popen(
                    cmd, cwd=plan_path.parent, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                )
                for line in proc.stdout:
                    text = line.decode(errors="replace")
                    sys.stdout.write(text)
                    lf.write(text)
                proc.wait()
            result = proc
        else:
            result = subprocess.run(cmd, cwd=plan_path.parent, env=env)
        if result.returncode != 0:
            logger.error("%s%s[%d/%d] FAILED — %s (exit %d)%s",
                         _RED, _BOLD, i, len(experiments), run_label, result.returncode, _R)
            failed.append(run_label)
        else:
            logger.info("%s%s[%d/%d] OK — %s%s",
                        _GREEN, _BOLD, i, len(experiments), run_label, _R)

    logger.info("=" * 60)
    if dry_run:
        logger.info("Dry run complete.")
        return

    if debug:
        # ── Debug summary — ready / not ready for full run ─────────────────
        _banner("DEBUG RUN SUMMARY", _YELL)
        all_passed = True
        for exp_cfg in experiments:
            hypothesis_id = exp_cfg["hypothesis_id"]
            experiment    = exp_cfg["experiment"]
            fold          = exp_cfg["fold"]
            own_keys      = exp_cfg.get("_exp_own_keys", [])
            extras        = {k: exp_cfg[k] for k in own_keys}
            run_label     = f"{hypothesis_id} / {experiment} fold{fold}" + (
                "  [" + "  ".join(f"{k}={v}" for k, v in sorted(extras.items())) + "]"
                if extras else ""
            )
            if run_label in failed:
                status = f"{_RED}{_BOLD}NOT READY{_R}"
                all_passed = False
            else:
                status = f"{_GREEN}{_BOLD}READY    {_R}"
            logger.info("  %s  %s", status, run_label)
        logger.info("")
        if all_passed:
            logger.info("%s%s  All experiments passed debug — safe to run full pipeline.%s",
                        _GREEN, _BOLD, _R)
        else:
            logger.info("%s%s  Some experiments failed — fix errors before full run.%s",
                        _RED, _BOLD, _R)
        return

    if failed:
        logger.error("Failed: %s", failed)
    else:
        logger.info("All %d experiment(s) completed.", len(experiments))

    subprocess.run([sys.executable, str(Path(__file__).parent / "research_report.py")])


if __name__ == "__main__":
    fire.Fire(main)
