# Report build system
# Usage:
#   make week_01        compile week 1  ->  pdfs/week_01.pdf
#   make week_02        compile week 2  ->  pdfs/week_02.pdf
#   make all            compile all weeks
#   make clean          remove aux files

WEEKS_DIR = weeks
PDFS_DIR  = pdfs
PDFLATEX  = pdflatex -interaction=nonstopmode -halt-on-error

$(PDFS_DIR):
	mkdir -p $(PDFS_DIR)

week_%: $(PDFS_DIR)
	cd $(WEEKS_DIR) && $(PDFLATEX) week_$*.tex
	cd $(WEEKS_DIR) && $(PDFLATEX) week_$*.tex
	mv $(WEEKS_DIR)/week_$*.pdf $(PDFS_DIR)/week_$*.pdf
	cd $(WEEKS_DIR) && rm -f week_$*.aux week_$*.log week_$*.out week_$*.toc

all: $(PDFS_DIR)
	@for f in $(WEEKS_DIR)/week_[0-9][0-9].tex; do \
		n=$$(basename $$f .tex | sed 's/week_//'); \
		$(MAKE) week_$$n; \
	done

clean:
	rm -f $(WEEKS_DIR)/*.aux $(WEEKS_DIR)/*.log $(WEEKS_DIR)/*.out $(WEEKS_DIR)/*.toc

.PHONY: all clean
