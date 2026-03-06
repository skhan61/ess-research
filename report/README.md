# Report

Weekly research report for the ESS Research project.

## File

- `weekly_report.tex` — main LaTeX source (self-contained, all figures linked from `../outputs/`)
- `Makefile` — build helper

## To compile

**Install TeX Live** (Ubuntu):
```bash
sudo apt install texlive-latex-extra texlive-fonts-recommended
```

**Then build:**
```bash
cd report/
make
# → weekly_report.pdf
```

Or manually:
```bash
pdflatex -interaction=nonstopmode weekly_report.tex
pdflatex -interaction=nonstopmode weekly_report.tex   # second pass for cross-refs
```

## Contents

1. Project overview — dataset, model architecture, training config
2. Experimental plan — 5 hypotheses (H1–H5)
3. Results per hypothesis — tables + prediction images
4. Consolidated summary table vs SOTA
5. Discussion — why LoRA works, why box > point, domain gap analysis
6. Future research directions — H6–H10 proposals
7. Reproducibility notes

## Updating

When new experiments complete, update the results tables in `weekly_report.tex`
and add new figure references pointing to `../outputs/Hx/...`.