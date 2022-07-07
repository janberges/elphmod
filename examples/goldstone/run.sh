#!/bin/bash

# Copyright (C) 2017-2022 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

url=http://www.pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard # [1, 2]
pp=N.upf
test -e $pp || (wget $url/$pp.gz && gunzip $pp)

# [1] van Setten et al., Comput. Phys. Commun. 226, 39 (2018)
# [2] Hamann, Phys. Rev. B 88, 085117 (2013)

mpirun pw.x < pw.in | tee pw.out

for method in dfpt cdfpt
do
    mpirun ph.x -ndiag 1 < $method.in | tee $method.out
    echo "&INPUT fildyn='$method.dyn' flfrc='$method.ifc' /" | mpirun -n 1 q2r.x
    cp -rT work/_ph0/N2.phsave $method.phsave
done

mpirun python3 goldstone.py
