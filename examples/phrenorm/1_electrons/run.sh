#!/bin/bash

PP=Be.pz-hgh.UPF

test -e $PP || wget https://www.quantum-espresso.org/upf_files/$PP

mpirun -np 2 pw.x -nk 2 < Be.scf
mpirun -np 2 pw.x -nk 2 < Be.bands
mpirun -np 2 bands.x -nk 2 < bands.in

echo "plot 'bands.dat.gnu' with lines" | gnuplot -p
