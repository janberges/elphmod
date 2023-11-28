.PHONY: all clean

w = 12
h = 9
in = 2.54

all: elphmod.png elphmod_square.png elphmod_banner.png

%.pdf: %.tex
	pdflatex --interaction=batchmode $*

elphmod.png: elphmod.pdf
	convert -density `perl -e "print 640 / ($(w) / $(in))"` $< \
        -flatten PNG8:$@

elphmod_square.png: elphmod.pdf
	convert -density `perl -e "print 640 / ($(w) / $(in))"` $< \
        -gravity center -extent 640x640 PNG8:$@

elphmod_banner.png: elphmod.pdf
	convert -density `perl -e "print 480 / ($(h) / $(in))"` $< \
        -gravity center -extent 1280x640 PNG8:$@

clean:
	rm -f *.aux *.log *.pdf *.png
