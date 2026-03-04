# ESS Research — SAM3 + LoRA for Surgical Instrument Segmentation

Fine-tuning [SAM3](https://huggingface.co/facebook/sam3) with LoRA adapters for
automatic surgical instrument segmentation in endoscopic sinus surgery (ESS) video.

**Dataset:** UW-Sinus-Surgery-C/L — 4345 cadaver frames + 4658 live frames
**Baseline to beat:** HFRF-Net Dice = 0.9374
**Primary metric:** `test/dice`

---

## What We Are Testing

All experiments are defined in [`research_plan.yaml`](research_plan.yaml).

### H1 — Zero-shot SAM3, Cadaver (C→C)

Establishes a zero-shot baseline on cadaver data.
Expected: high recall, low precision (over-segmentation).

### H2 — Zero-shot SAM3, Live (L→L, 3-fold CV)

Zero-shot on live surgical video. Live data is harder than cadaver
(blood, smoke, motion blur). Expected Dice lower than H1.

### H3 — LoRA Fine-tuning, Within-domain (C→C + L→L)

LoRA adapters injected into SAM3's ViT attention projections (`q_proj`, `v_proj`).
Fine-tuned separately on cadaver and live data to correct zero-shot over-segmentation.
Target: Dice > 0.80 on C→C, Dice > 0.70 mean on L→L.

### H4 — Prompt Type Ablation, LoRA C→C

SAM3 supports three prompt types. H3 uses text only.
H4 isolates each type on C→C to measure their individual contribution:

| Prompt | Input |
| ------ | ----- |
| `text` | `"surgical instrument"` (H3 baseline) |
| `box` | GT bounding box `[x1, y1, x2, y2]` |
| `point` | GT centre-of-mass `[cx, cy]` |
| `all` | text + box + point combined |

Expected ranking: all > box > point > text.

---

## Installation

Requires Python ≥ 3.10 and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/skhan61/ess-research.git
cd ess-research
uv sync
```

Add a `.env` file with your HuggingFace token (needed to download SAM3 weights):

```text
HF_TOKEN=hf_...
```

Place the dataset at:

```text
data/uw-sinus-surgery-CL/
  cadaver/images/*.jpg
  cadaver/labels/*.png
  live/images/*.jpg
  live/labels/*.png
```

---

## Running All Experiments

```bash
uv run python scripts/run_all_experiments.py --plan research_plan.yaml
```

This runs all hypotheses in order. Already-completed experiments are auto-skipped.
A summary report is generated automatically when all runs finish.

To preview what will run without executing:

```bash
uv run python scripts/run_all_experiments.py --plan research_plan.yaml --dry_run True
```

---

## Outputs

Each run saves to `outputs/{HX}/{experiment}_fold{N}/`:

```text
metrics.csv          — train loss (per step) + val Dice/IoU/loss (per epoch)
checkpoints/         — best checkpoint (monitored by val/dice)
predictions/         — visualised predictions on vis_samples frames
```

To print a results summary at any time:

```bash
uv run python scripts/summarize_results.py
```
