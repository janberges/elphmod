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

mpirun -n $NP pw.x -nk $NK < pw.in | tee pw.out
mpirun -n $NP ph.x -nk $NK < ph.in | tee ph.out

ph2epw

mpirun -n $NP pw.x -nk $NK < nscf.in | tee nscf.out
mpirun -n $NK epw.x -nk $NK < epw.in | tee epw.out

mpirun -n 1 python3 ph_vs_epw.py
