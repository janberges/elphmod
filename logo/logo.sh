#!/bin/bash

w=12
h=9
in=2.54

pdflatex --interaction=batchmode elphmod

convert -density `perl -e "print 640 / ($w / $in)"` elphmod.pdf \
    -flatten PNG8:elphmod.png

convert -density `perl -e "print 640 / ($w / $in)"` elphmod.pdf \
    -background white -gravity center -extent 640x640 PNG8:elphmod_square.png

convert -density `perl -e "print 480 / ($h / $in)"` elphmod.pdf \
    -background white -gravity center -extent 1280x640 PNG8:elphmod_banner.png
