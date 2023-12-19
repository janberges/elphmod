.PHONY: all clean

all: elphmod.pdf elphmod.png elphmod_square.png elphmod_banner.png

%.pdf: %.tex
	pdflatex --interaction=batchmode $*

elphmod.png: elphmod.svg
	inkscape -w 480 -b white -o $@ $<
	python3 optimize.py $@ $@

elphmod_square.png: elphmod.png
	convert $< -gravity center -extent 480x480 $@

elphmod_banner.png: elphmod.png
	convert $< -gravity center -extent 1280x640 $@

clean:
	rm -f *.aux *.log *.pdf *_*.png
