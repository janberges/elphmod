#!/bin/bash

# Copyright (C) 2017-2025 elphmod Developers
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
mpirun pw.x -nk $nk < nscf.in | tee nscf.out
mpirun projwfc.x -nk $nk < projwfc.in | tee projwfc.out

mpirun pw.x -nk $nk < scf.in | tee scf.out

for method in dfpt cdfpt
do
    mpirun ph.x -nk $nk < $method.in | tee $method.out

    fildyn=$method.dyn dvscf_dir=$method.save ph2epw
done

mpirun pw.x -nk $nk < nscf.in | tee nscf.out

for method in dfpt cdfpt
do
    mpirun -n $nk epw.x -nk $nk < epw-$method.in | tee epw-$method.out

    mv work/graphene.epmatwp $method.epmatwp
done

mpirun python3 phrenorm_graphene.py
