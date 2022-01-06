#!/bin/bash

# Copyright (C) 2017-2022 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

url=http://www.pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard # [1, 2]
pp=C.upf
test -e $pp || (wget $url/$pp.gz && gunzip $pp)

# [1] van Setten et al., Comput. Phys. Commun. 226, 39 (2018)
# [2] Hamann, Phys. Rev. B 88, 085117 (2013)

nk=2

mpirun pw.x -nk $nk < scf.in | tee scf.out
mpirun pw.x -nk $nk < nscf.in | tee nscf.out
mpirun projwfc.x -nk $nk < projwfc.in | tee projwfc.out

for method in dfpt cdfpt
do
    mpirun pw.x -nk $nk < scf.in | tee scf.out
    mpirun ph.x -nk $nk < $method.in | tee $method.out

    mpirun q2r.x < q2r.in | tee q2r.out
    ../../bin/ph2epw

    mpirun pw.x -nk $nk < nscf.in | tee nscf.out
    mpirun -n $nk epw.x -nk $nk < epw.in | tee epw.out

    mv ifc $method.ifc
    mv work/graphene.epmatwp $method.epmatwp
done

mpirun python3 phrenorm_graphene.py
