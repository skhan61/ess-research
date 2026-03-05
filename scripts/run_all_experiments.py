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
    # Keys that are handled internally by the runner — never forwarded to main.py
    # and never included in run_name / variant suffix.
    _RUNNER_KEYS = {"experiment", "fold", "source_hypothesis"}

    configs: list[dict[str, object]] = []
    for hyp in plan.get("hypotheses", []):
        hyp_cfg: dict[str, object] = hyp.get("config", {})   # per-hypothesis overrides
        for exp in hyp.get("experiments", []):
            entry = {**hyp_cfg, **exp}            # exp keys override hyp_cfg
            entry["hypothesis_id"] = hyp["id"]   # e.g. "H2"
            # Keys defined directly on this experiment entry that affect run naming
            entry["_exp_own_keys"] = [k for k in exp if k not in _RUNNER_KEYS]
            # source_hypothesis: which prior hypothesis' checkpoint dir to load from
            entry["_source_hypothesis"] = exp.get("source_hypothesis")  # None if absent
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


def _find_resume_checkpoint(ckpt_dir: Path) -> Path | None:
    """Return last.ckpt if it exists — used to resume interrupted training."""
    last = ckpt_dir / "last.ckpt"
    return last if last.exists() else None


def _build_cmd(exp_cfg: dict[str, object], global_cfg: dict[str, object],
               run_name: str | None = None,
               ckpt_path: Path | None = None,
               resume_ckpt_path: Path | None = None) -> list[str]:
    """
    Merge per-experiment keys (experiment, fold) with global config and
    build the ``python main.py`` subprocess command.

    run_name:         passed as --run_name_override so main.py uses the same
                      variant-aware directory name as the runner (e.g. includes prompt_mode).
    ckpt_path:        passed as --ckpt_path  → test-only mode in main.py
    resume_ckpt_path: passed as --resume_ckpt_path → resumes interrupted training
    """
    merged = {**global_cfg, **exp_cfg}   # exp keys override global if they clash
    tokens = [sys.executable, "main.py"]
    for k, v in merged.items():
        tokens += [f"--{k}", str(v)]
    if run_name is not None:
        tokens += ["--run_name_override", run_name]
    if ckpt_path is not None:
        tokens += ["--ckpt_path", str(ckpt_path)]
    if resume_ckpt_path is not None:
        tokens += ["--resume_ckpt_path", str(resume_ckpt_path)]
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
        debug_root.mkdir(parents=True, exist_ok=True)
        # Always start with a fresh runner.log but preserve completed experiment
        # outputs so already-passing debug experiments are skipped on re-run.
        log_file = debug_root / "runner.log"
        if log_file.exists():
            log_file.unlink()
        setup_logging(log_file=log_file)
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
        # In debug mode use outputs/debug/ so real outputs are never touched.
        if debug:
            hyp_save_dir = str(Path("outputs/debug") / hypothesis_id)
            merged_cfg = {**global_cfg, "save_dir": hyp_save_dir,
                          "max_epochs": 1,
                          "limit_train_batches": 2,
                          "limit_val_batches": 2,
                          "limit_test_batches": 2}
        else:
            hyp_save_dir = str(Path(global_cfg["save_dir"]) / hypothesis_id)
            merged_cfg = {**global_cfg, "save_dir": hyp_save_dir}

        metrics_path = Path(hyp_save_dir) / run_name / "metrics.csv"

        if _has_test_results(metrics_path):
            logger.info("[%d/%d] SKIP — %s (results exist: %s)", i, len(experiments), run_label, metrics_path)
            continue

        # Strip internal bookkeeping keys before calling main.py
        _INTERNAL = {"hypothesis_id", "_exp_own_keys", "_source_hypothesis", "source_hypothesis"}
        main_cfg = {k: v for k, v in exp_cfg.items() if k not in _INTERNAL}

        source_hyp = exp_cfg.get("_source_hypothesis")  # e.g. "H3", or None

        ckpt_path: Path | None = None
        if source_hyp:
            # H5-style: load checkpoint from a prior hypothesis' output dir.
            # In debug mode look in outputs/debug/{source_hyp}/; in full run
            # look in the real save_dir. H1-H4 debug runs execute before H5 so
            # the debug checkpoints already exist when H5 experiments start.
            if debug:
                src_ckpt_dir = Path("outputs/debug") / source_hyp / run_name / "checkpoints"
            else:
                src_ckpt_dir = Path(global_cfg["save_dir"]) / source_hyp / run_name / "checkpoints"
            ckpt_path = _find_best_checkpoint(src_ckpt_dir)
            if ckpt_path is None:
                logger.warning(
                    "[%d/%d] SKIP (no checkpoint in %s) — %s",
                    i, len(experiments), src_ckpt_dir, run_label,
                )
                continue
            _banner(
                f"[{i}/{len(experiments)}] TEST-ONLY [{source_hyp}]  {run_label}"
                f"  ckpt={ckpt_path.name}",
                _CYAN,
            )
        elif test_only:
            ckpt_dir = Path(hyp_save_dir) / run_name / "checkpoints"
            ckpt_path = _find_best_checkpoint(ckpt_dir)
            if ckpt_path is None:
                logger.warning("[%d/%d] SKIP (test_only, no checkpoint) — %s", i, len(experiments), run_label)
                continue
            _banner(f"[{i}/{len(experiments)}] TEST-ONLY  {run_label}  ckpt={ckpt_path.name}", _CYAN)
        else:
            # Check for an interrupted previous run — resume from last.ckpt if found.
            # Only applies to training experiments (not debug, not source_hyp, not test_only).
            resume_ckpt: Path | None = None
            if not debug and not source_hyp:
                resume_ckpt_dir = Path(hyp_save_dir) / run_name / "checkpoints"
                resume_ckpt = _find_resume_checkpoint(resume_ckpt_dir)
                if resume_ckpt:
                    logger.info(
                        "%s%s[%d/%d] RESUME from %s — %s%s",
                        _YELL, _BOLD, i, len(experiments), resume_ckpt.name, run_label, _R,
                    )
                else:
                    _banner(f"[{i}/{len(experiments)}] START  {run_label}", _CYAN)
            else:
                _banner(f"[{i}/{len(experiments)}] START  {run_label}", _CYAN)

        cmd = _build_cmd(main_cfg, merged_cfg, run_name=run_name,
                         ckpt_path=ckpt_path,
                         resume_ckpt_path=resume_ckpt if not source_hyp else None)
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
            # Brief pause so the OS fully reclaims GPU memory before the next
            # subprocess loads SAM3 (~11 GB). Without this, back-to-back
            # experiments can OOM even though each process exits cleanly.
            import time; time.sleep(5)
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
