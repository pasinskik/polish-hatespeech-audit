PAPER = paper

.PHONY: all clean view watch

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	tectonic $(PAPER).tex

clean:
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).bbl $(PAPER).blg $(PAPER).out $(PAPER).toc

view: $(PAPER).pdf
	open $(PAPER).pdf

watch:
	tectonic -X watch $(PAPER).tex
