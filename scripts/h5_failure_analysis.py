"""
scripts/h5_failure_analysis.py
Generate H5 failure analysis figures for the weekly report:
  1. Per-run Dice distribution box plot (all H5 runs)
  2. Worst-performing frames table (text prompt runs, Dice < 0.5)
  3. Worst-frame image grid (top-N lowest Dice images)

Outputs → outputs/H5/analysis/
Run:
    uv run python scripts/h5_failure_analysis.py
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO    = Path(__file__).parent.parent
H5_DIR  = REPO / "outputs" / "H5"
OUT_DIR = REPO / "outputs" / "H5" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Load all per_image_dice.csv ───────────────────────────────

def load_all_runs():
    runs = {}
    for csv_path in sorted(H5_DIR.glob("*/predictions/per_image_dice.csv")):
        run_name = csv_path.parent.parent.name
        rows = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                rows.append({
                    "stem":       row["stem"],
                    "video_id":   row["video_id"],
                    "image_path": row["image_path"],
                    "dice":       float(row["dice"]),
                })
        runs[run_name] = rows
    return runs


# ── 1. Dice distribution plots ───────────────────────────────

def plot_distributions(runs: dict):
    groups = {"text (H3)": [], "box": [], "point": [], "all": []}
    for run_name, rows in runs.items():
        dices = [r["dice"] for r in rows]
        if   "prompt_modebox"   in run_name: groups["box"].extend(dices)
        elif "prompt_modepoint" in run_name: groups["point"].extend(dices)
        elif "prompt_modeall"   in run_name: groups["all"].extend(dices)
        else:                                groups["text (H3)"].extend(dices)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Box plot by prompt type
    keys   = ["text (H3)", "point", "box", "all"]
    colors = ["#607D8B", "#1565C0", "#2E7D32", "#795548"]
    bp = axes[0].boxplot([groups[k] for k in keys], patch_artist=True,
                         medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    axes[0].set_xticklabels(keys)
    axes[0].set_ylabel("Dice score")
    axes[0].set_title("Per-image Dice by prompt type (all H5 folds)")
    axes[0].axhline(0.9374, color="red",    linestyle="--", linewidth=1.2, label="HFRF-Net SOTA")
    axes[0].axhline(0.5,    color="orange", linestyle=":",  linewidth=1.2, label="Failure threshold")
    axes[0].legend(fontsize=8); axes[0].set_ylim(0, 1.02)

    # Histogram of text runs
    td = groups["text (H3)"]
    axes[1].hist(td, bins=30, color="#607D8B", alpha=0.8, edgecolor="white")
    axes[1].axvline(float(np.mean(td)), color="black",  linestyle="-",  linewidth=1.5,
                    label=f"Mean={np.mean(td):.3f}")
    axes[1].axvline(0.9374,             color="red",    linestyle="--", linewidth=1.2, label="HFRF-Net SOTA")
    axes[1].axvline(0.5,                color="orange", linestyle=":",  linewidth=1.2, label="Failure threshold")
    axes[1].set_xlabel("Dice score"); axes[1].set_ylabel("Frame count")
    axes[1].set_title("Dice histogram — text-prompt runs (all folds)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    out = OUT_DIR / "dice_distributions.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved: {out}")
    return out


# ── 2. Worst-frames image grid ────────────────────────────────

def plot_worst_grid(runs: dict, threshold=0.5, n=10):
    worst = []
    for run_name, rows in runs.items():
        if any(p in run_name for p in ["box", "point", "all"]):
            continue  # text runs only
        for r in rows:
            if r["dice"] < threshold:
                worst.append((r["dice"], run_name, r))
    worst.sort(key=lambda x: x[0])

    valid = []
    for _, run_name, r in worst:
        img_path = REPO / r["image_path"]
        if img_path.exists():
            valid.append((img_path, r, run_name))
        if len(valid) >= n:
            break

    if not valid:
        print(f"  No frames with Dice < {threshold} found on disk.")
        return None

    cols = min(5, len(valid))
    nrows = (len(valid) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 3, nrows * 3))
    axes = np.array(axes).reshape(-1)
    for ax in axes: ax.axis("off")

    for idx, (img_path, r, run_name) in enumerate(valid):
        img = Image.open(img_path).convert("RGB")
        axes[idx].imshow(img)
        axes[idx].set_title(f"{r['stem']}\nDice={r['dice']:.3f}", fontsize=7)
        axes[idx].axis("off")

    plt.suptitle(f"H5: Worst frames (text prompt, Dice < {threshold})", fontsize=10)
    plt.tight_layout()
    out = OUT_DIR / "worst_frames_grid.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved: {out}")
    return out, worst


# ── Main ──────────────────────────────────────────────────────

# ── Direct comparison table (LaTeX) ─────────────────────────

def generate_direct_comparison_tex(summary_csv: Path, hfrfnet_dice=0.9374) -> str:
    """Read summary.csv and produce a LaTeX table comparing
    SAM3+LoRA L->L results vs HFRF-Net on the same live dataset."""
    rows = []
    with open(summary_csv) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # H3 L->L text rows
    h3_live = [(r["fold"], float(r["dice"]))
               for r in rows
               if r["hypothesis"] == "H3" and r["experiment"] == "L->L"
               and r["variant"] == ""]
    h3_live.sort(key=lambda x: int(x[0]))

    # H4 L->L box rows
    h4_box = [(r["fold"], float(r["dice"]))
              for r in rows
              if r["hypothesis"] == "H4" and r["experiment"] == "L->L"
              and r["variant"] == "_prompt_modebox"]
    h4_box.sort(key=lambda x: int(x[0]))

    h3_mean = np.mean([d for _, d in h3_live])

    def delta(d): return f"+{d - hfrfnet_dice:.3f}" if d >= hfrfnet_dice else f"{d - hfrfnet_dice:.3f}"
    def color(d): return r"\textcolor{sota}" if d >= hfrfnet_dice else ""

    lines = [
        r"\begin{center}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Prompt} & \textbf{Dice (per fold)} & \textbf{$\Delta$ vs HFRF-Net} \\",
        r"\midrule",
        f"HFRF-Net (SOTA, UW-Sinus live) & --- & {hfrfnet_dice:.4f} & --- \\\\",
        r"\midrule",
    ]

    # H3 rows
    for i, (fold, dice) in enumerate(h3_live):
        label = "SAM3+LoRA (H3)" if i == 0 else ""
        prompt = "text" if i == 0 else ""
        lines.append(f"{label} & {prompt} & fold{fold}: {dice:.4f} & ${delta(dice)}$ \\\\")
    lines.append(f" & & \\textbf{{mean: {h3_mean:.4f}}} & ${delta(h3_mean)}$ \\\\")
    lines.append(r"\midrule")

    # H4 box rows
    for i, (fold, dice) in enumerate(h4_box):
        label = "SAM3+LoRA (H4)" if i == 0 else ""
        prompt = "box" if i == 0 else ""
        c = color(dice)
        if c:
            lines.append(f"{label} & {prompt} & fold{fold}: {c}{{\\textbf{{{dice:.4f}}}}} & {c}{{${delta(dice)}$}} \\\\")
        else:
            lines.append(f"{label} & {prompt} & fold{fold}: {dice:.4f} & ${delta(dice)}$ \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print("Loading per-image Dice CSVs from H5 ...")
    runs = load_all_runs()
    print(f"  {len(runs)} runs, "
          f"{sum(len(v) for v in runs.values())} total frames")

    print("\nPlotting Dice distributions ...")
    plot_distributions(runs)

    print("\nBuilding worst-frames grid ...")
    result = plot_worst_grid(runs, threshold=0.5, n=10)

    # Summary stats
    all_dices = [r["dice"] for rows in runs.values() for r in rows]
    print(f"\nSummary across all H5 runs:")
    print(f"  Frames total : {len(all_dices)}")
    print(f"  Mean Dice    : {np.mean(all_dices):.4f}")
    print(f"  Min Dice     : {np.min(all_dices):.4f}")
    print(f"  Max Dice     : {np.max(all_dices):.4f}")
    print(f"  Dice < 0.50  : {sum(d < 0.50 for d in all_dices)} frames")
    print(f"  Dice < 0.70  : {sum(d < 0.70 for d in all_dices)} frames")

    if result:
        _, worst = result
        print(f"\nWorst 10 frames (text prompt, Dice < 0.5):")
        print(f"  {'stem':<20} {'video':<8} {'run':<30} {'dice'}")
        for dice, run, r in worst[:10]:
            print(f"  {r['stem']:<20} {r['video_id']:<8} {run:<30} {dice:.4f}")

    print(f"\nAll outputs in: {OUT_DIR}")

    # Generate direct comparison LaTeX table
    summary_csv = REPO / "outputs" / "summary.csv"
    if summary_csv.exists():
        print("\nDirect comparison table (LaTeX):")
        tex = generate_direct_comparison_tex(summary_csv)
        print(tex)
        (OUT_DIR / "direct_comparison.tex").write_text(tex)
        print(f"Saved: {OUT_DIR / 'direct_comparison.tex'}")
