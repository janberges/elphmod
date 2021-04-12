#!/bin/bash

w=12
h=9
px=640
in=2.54

pdflatex --interaction=batchmode logo

convert -density `perl -e "print $px / ($w / $in)"` logo.pdf \
    -flatten PNG8:logo.png

convert -density `perl -e "print $px / ($w / $in)"` logo.pdf \
    -background white -gravity center -extent 640x640 PNG8:logo_square.png

convert -density `perl -e "print $px / ($h / $in)"` logo.pdf \
    -background white -gravity center -extent 1280x640 PNG8:logo_banner.png
