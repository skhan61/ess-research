# ESS Research — SAM3 + LoRA for Surgical Instrument Segmentation

Fine-tuning [SAM3](https://huggingface.co/facebook/sam3) with LoRA adapters for
automatic surgical instrument segmentation in endoscopic sinus surgery (ESS) video.

**Dataset:** UW-Sinus-Surgery-C/L — 4345 cadaver frames (S01–S10) + 4658 live frames (L01–L03)
**Baseline:** HFRF-Net Dice = 0.9374
**Primary metric:** `test/dice`

---

## Research Plan

| ID | Title | Prompt | Status |
| -- | ----- | ------ | ------ |
| H1 | Zero-shot SAM3 — Cadaver (C→C) | text | done |
| H2 | Zero-shot SAM3 — Live (L→L, 3-fold CV) | text | done |
| H3 | LoRA fine-tuning — Within-domain (C→C + L→L) | text | running |
| H4 | Prompt type ablation — LoRA C→C (box / point / all) | box / point / all | queued |

Config for every experiment lives in [`research_plan.yaml`](research_plan.yaml).
Edit the YAML — no code changes needed to add or modify experiments.

---

## Installation

Requires Python ≥ 3.10 and [uv](https://github.com/astral-sh/uv).

```bash
# Clone
git clone https://github.com/skhan61/ess-research.git
cd ess-research

# Install all dependencies (creates .venv automatically)
uv sync

# Place dataset at:
#   data/uw-sinus-surgery-CL/cadaver/images/*.jpg
#   data/uw-sinus-surgery-CL/cadaver/labels/*.png
#   data/uw-sinus-surgery-CL/live/images/*.jpg
#   data/uw-sinus-surgery-CL/live/labels/*.png
```

Add a `.env` file with your HuggingFace token so SAM3 weights can be downloaded:

```text
HF_TOKEN=hf_...
```

---

## Running Experiments

### Run all experiments in sequence (recommended)

```bash
uv run python scripts/run_all_experiments.py --plan research_plan.yaml
```

Already-completed experiments are auto-skipped (detected by presence of `test/dice` in `metrics.csv`).

### Dry-run (print commands without executing)

```bash
uv run python scripts/run_all_experiments.py --plan research_plan.yaml --dry_run True
```

### Run a single experiment manually

```bash
# LoRA fine-tuning, cadaver within-domain, text prompt
uv run python main.py \
  --experiment "C->C" --fold 1 \
  --data_root data/uw-sinus-surgery-CL \
  --image_size 336 --text_prompt "surgical instrument" \
  --batch_size 8 --num_workers 8 --pin_memory true \
  --max_epochs 50 --lr 0.0001 --loss combo \
  --early_stopping_patience 10 \
  --save_dir outputs/H3 --save_predictions true --vis_samples 8 \
  --log_level INFO --use_wandb false --wandb_project ess-research \
  --use_lora true --lora_rank 4 --lora_alpha 16.0 --lora_dropout 0.1 \
  --prompt_mode text --precision 16-mixed

# Zero-shot evaluation (no training)
uv run python main.py --experiment "C->C" --fold 1 ... --zero_shot true
```

### Prompt modes

The `--prompt_mode` flag controls which SAM3 prompts are used:

| Mode | What is passed to SAM3 |
| ---- | ---------------------- |
| `text` | Text string (`"surgical instrument"`) |
| `box` | GT bounding box `[x1, y1, x2, y2]` |
| `point` | GT centre-of-mass `[cx, cy]` |
| `all` | Text + box + point combined |

---

## Project Structure

```text
ess-research/
├── main.py                        # Training entry point
├── research_plan.yaml             # Single source of truth for all experiments
├── src/
│   ├── model/
│   │   ├── sam3/model.py          # SAM3 wrapper + LoRA injection + prompt routing
│   │   └── module.py              # Lightning training/val/test step
│   ├── datamodule/
│   │   ├── datamodule.py          # C→C / L→L / C→L / L→C splits
│   │   └── dataset.py             # Image + mask + text/box/point prompts
│   ├── losses/                    # BCE, Dice, combo, focal losses
│   └── metrics/                   # Dice, IoU, precision, recall
├── scripts/
│   ├── run_all_experiments.py     # Sequential experiment runner
│   ├── summarize_results.py       # Print results table
│   └── research_report.py        # Generate report after all runs
└── outputs/
    └── {HX}/{experiment}_fold{N}/
        ├── metrics.csv            # Per-step train loss + per-epoch val metrics
        ├── checkpoints/best-*.ckpt
        └── predictions/           # Visualised predictions (vis_samples frames)
```

---

## Results

Results are saved to `outputs/{HX}/{experiment}_fold{N}/metrics.csv`.
To print a summary table:

```bash
uv run python scripts/summarize_results.py
```
