#!/bin/bash

# Copyright (C) 2017-2026 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

eval `elphmodenv`

echo 'Using Hartwigsen-Goedecker-Hutter pseudopotentials'
echo '[1] Hartwigsen et al., Phys. Rev. B 58, 3641 (1998)'
echo '[2] Goedecker et al., Phys. Rev. B 54, 1703 (1996)'

url=https://pseudopotentials.quantum-espresso.org/upf_files

for pp in S.pbe-hgh.UPF Ta.pbe-hgh.UPF
do
    test -e $pp || wget $url/$pp
done

: ${NP:=2}
: ${NK:=2}

mpirun -n $NP pw.x -nk $NK < scf.in | tee scf.out

for method in dfpt cdfpt
do
    mpirun -n $NP ph.x -nk $NK < $method.in | tee $method.out

    fildyn=$method.dyn dvscf_dir=$method.save ph2epw
done

mpirun -n $NP pw.x -nk $NK < nscf.in | tee nscf.out

for method in dfpt cdfpt
do
    mpirun -n $NK epw.x -nk $NK < epw-$method.in | tee epw-$method.out

    mv work/TaS2.epmatwp $method.epmatwp
done

mpirun -n $NP python3 phrenorm.py
mpirun -n $NP python3 defpot.py
mpirun -n $NP python3 decay.py
