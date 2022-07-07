#!/bin/bash

# Copyright (C) 2017-2022 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

url=https://pseudopotentials.quantum-espresso.org/upf_files

for pp in S.pbe-hgh.UPF Ta.pbe-hgh.UPF
do
    test -e $pp || wget $url/$pp
done

nk=2

for method in dfpt cdfpt
do
    mpirun pw.x -nk $nk < scf.in | tee scf.out
    mpirun ph.x -nk $nk < $method.in | tee $method.out

    mpirun q2r.x < q2r-$method.in | tee q2r-$method.out
    fildyn=$method.dyn dvscf_dir=$method.save ../../bin/ph2epw

    mpirun pw.x -nk $nk < nscf.in | tee nscf.out
    mpirun -n $nk epw.x -nk $nk < epw-$method.in | tee epw-$method.out

    mv work/TaS2.epmatwp $method.epmatwp
done

mpirun python3 phrenorm.py
mpirun python3 defpot.py
mpirun python3 decay.py
