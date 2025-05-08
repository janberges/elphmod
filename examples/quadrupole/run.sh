#!/bin/bash

# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

eval `elphmodenv`

echo 'Using normconserving pseudopotentials from PseudoDojo'
echo '[1] van Setten et al., Comput. Phys. Commun. 226, 39 (2018)'
echo '[2] Hamann, Phys. Rev. B 88, 085117 (2013)'

gh=https://raw.githubusercontent.com/PseudoDojo/ONCVPSP-PBE-SR/refs/heads/master

for X in Ta S
do
    test -e $X.upf || (wget $gh/$X/${X}_std.upf && mv ${X}_std.upf $X.upf)
done

nk=2

mpirun pw.x -nk $nk < scf.in | tee scf.out
mpirun ph.x -nk $nk < ph.in | tee ph.out

ph2epw

mpirun pw.x -nk $nk < nscf.in | tee nscf.out
mpirun -n $nk epw.x -nk $nk < epw.in | tee epw.out

mpirun pw.x -nk $nk < scf.in | tee scf.out
mpirun ph.x -nk $nk < phref.in | tee phref.out

mpirun python3 quadrupole.py
