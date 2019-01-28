#!/bin/bash

for PP in S.pbe-hgh.UPF Ta.pbe-hgh.UPF
do
    test -e $PP || wget https://www.quantum-espresso.org/upf_files/$PP
done

mpirun -np 4 pw.x -nk 2 < TaS2.scf
mpirun -np 4 pw.x -nk 2 < TaS2.bands
mpirun -np 4 bands.x -nk 2 < TaS2.pltbnd

mpirun -np 4 pw.x -nk 2 < TaS2.nscf
wannier90.x -pp TaS2
mpirun -np 4 pw2wannier90.x < TaS2.pw2w90
wannier90.x TaS2

echo "p 'bands.dat.gnu' u (\$1*2*pi/3.387):2, 'TaS2_band.dat' w l" | gnuplot -p

mpirun -np 4 ph.x -nk 2 < TaS2.dfpt
