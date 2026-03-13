# ESS Research — Claude Instructions

## Project Rules
- Never suggest running `main.py` directly — yaml runner is the only entry point
- Entry point: `uv run python scripts/run_all_experiments.py --plan research_plan.yaml`
- Debug run: add `--debug` flag
- All experiments defined in `research_plan.yaml`

## Key Technical Notes
- PyTorch 2.6: add_safe_globals([pathlib.PosixPath]) for checkpoint loading
- SAM3 always requires input_ids (text) — no text-free mode
- Spatial prompts: bypass processor, normalise coords manually (/ image_size)
- image_size must be divisible by 14 (ViT patch size)

## Report Conventions
- LaTeX reports in `report/weeks/week_NN.tex`
- References in `report/references.bib` (BibTeX only — no manual refs)
- Compile: `cd report && make week_NN`
- **Figures: EPS only — never PNG or JPG in LaTeX**
- Tone: scientific, table-driven, results-focused (follow week_01.tex style)
- No supervisor/email sections in reports
- No fake experiments — only document what was actually run

## GitHub
- Remote: https://github.com/skhan61/ess-research.git
- Push via: `git remote set-url origin https://<token>@github.com/skhan61/ess-research.git`
