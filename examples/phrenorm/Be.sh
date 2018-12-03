#!/bin/bash

PP=Be.pz-hgh.UPF

test -e $PP || wget https://www.quantum-espresso.org/upf_files/$PP

mpirun -np 2 pw.x -nk 2 < Be.scf
mpirun -np 2 pw.x -nk 2 < Be.bands
mpirun -np 2 bands.x -nk 2 < bands.in

mpirun -np 2 pw.x -nk 2 < Be.nscf
wannier90.x -pp Be
mpirun -np 2 pw2wannier90.x < Be.pw2w90
wannier90.x Be

echo "p 'bands.dat.gnu' u (\$1*2*pi/2.07):2, 'Be_band.dat' w l" | gnuplot -p

mpirun -np 2 ph.x -nk 2 < Be.dfpt
