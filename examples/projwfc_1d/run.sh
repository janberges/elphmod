#!/bin/bash

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

eval `elphmodenv`

echo 'Using normconserving pseudopotentials from PseudoDojo'
echo '[1] van Setten et al., Comput. Phys. Commun. 226, 39 (2018)'
echo '[2] Hamann, Phys. Rev. B 88, 085117 (2013)'

url=http://www.pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard
pp=C.upf
test -e $pp || (wget $url/$pp.gz && gunzip $pp)

nk=2

mpirun pw.x -nk $nk < scf.in | tee scf.out
mpirun pw.x -nk $nk < bands.in | tee bands.out
mpirun projwfc.x -nk $nk < projwfc.in | tee projwfc.out

mpirun pw.x -nk $nk < nscf.in | tee nscf.out
mpirun -n 1 wannier90.x -pp C
mpirun pw2wannier90.x < pw2w90.in | tee pw2w90.out
mpirun -n 1 wannier90.x C

mpirun python3 projwfc_1d.py
