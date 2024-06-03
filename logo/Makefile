.PHONY: all clean

all: elphmod.pdf elphmod.png elphmod_square.png elphmod_banner.png

%.pdf: %.tex
	pdflatex --interaction=batchmode $*

elphmod.png: elphmod.svg
	inkscape -w 480 -b white -o $@ $<
	python3 optimize.py $@ $@

elphmod_square.png elphmod_banner: elphmod.png
	python3 pad.py

clean:
	rm -f *.aux *.log *.pdf *_*.png
