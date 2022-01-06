#!/bin/bash

# Copyright (C) 2017-2022 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

for pp in S.pbe-hgh.UPF Ta.pbe-hgh.UPF
do
    test -e $pp || wget https://www.quantum-espresso.org/upf_files/$pp
done

nk=2

for method in dfpt cdfpt
do
    mpirun pw.x -nk $nk < scf.in | tee scf.out
    mpirun ph.x -nk $nk < $method.in | tee $method.out

    mpirun q2r.x < q2r.in | tee q2r.out
    ../../bin/ph2epw

    mpirun pw.x -nk $nk < nscf.in | tee nscf.out
    mpirun -n $nk epw.x -nk $nk < epw.in | tee epw.out

    mv ifc $method.ifc
    mv work/TaS2.epmatwp $method.epmatwp
done

mpirun python3 phrenorm.py
mpirun python3 defpot.py
mpirun python3 supercell_elph.py
mpirun python3 decay.py
