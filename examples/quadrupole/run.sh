#!/bin/bash

# Copyright (C) 2017-2026 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

eval `elphmodenv`

echo 'Using normconserving pseudopotentials from PseudoDojo'
echo '[1] van Setten et al., Comput. Phys. Commun. 226, 39 (2018)'
echo '[2] Hamann, Phys. Rev. B 88, 085117 (2013)'

url=http://www.pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard # [1, 2]

for pp in Ta.upf S.upf
do
    test -e $pp || (wget $url/$pp.gz && gunzip $pp)
done

: ${NP:=2}
: ${NK:=2}

mpirun -n $NP pw.x -nk $NK < scf.in | tee scf.out
mpirun -n $NP ph.x -nk $NK < ph.in | tee ph.out

ph2epw

mpirun -n $NP pw.x -nk $NK < nscf.in | tee nscf.out
mpirun -n $NK epw.x -nk $NK < epw.in | tee epw.out

mpirun -n $NP pw.x -nk $NK < scf.in | tee scf.out
mpirun -n $NP ph.x -nk $NK < phref.in | tee phref.out

mpirun -n $NP python3 quadrupole.py
