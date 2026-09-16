#!/bin/bash

# Copyright (C) 2017-2026 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

eval `elphmodenv`

echo 'Using normconserving pseudopotentials from PseudoDojo'
echo '[1] van Setten et al., Comput. Phys. Commun. 226, 39 (2018)'
echo '[2] Hamann, Phys. Rev. B 88, 085117 (2013)'

url=http://www.pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard
pp=C.upf
test -e $pp || (wget $url/$pp.gz && gunzip $pp)

: ${NP:=2}
: ${NK:=2}

mpirun -n $NP pw.x -nk $NK < scf.in | tee scf.out
mpirun -n $NP pw.x -nk $NK < bands.in | tee bands.out
mpirun -n $NP projwfc.x -nk $NK < projwfc.in | tee projwfc.out

mpirun -n $NP pw.x -nk $NK < nscf.in | tee nscf.out
mpirun -n 1 wannier90.x -pp C
mpirun -n $NP pw2wannier90.x -pd true < pw2w90.in | tee pw2w90.out
mpirun -n 1 wannier90.x C

mpirun -n $NP python3 projwfc_1d.py
