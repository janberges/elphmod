#!/bin/bash

# Copyright (C) 2017-2024 elphmod Developers
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

nk=2

mpirun pw.x -nk $nk < scf.in | tee scf.out
mpirun pw.x -nk $nk < nscf.in | tee nscf.out

for seedname in ws_yes ws_no
do
    mpirun -n 1 wannier90.x -pp $seedname
    mpirun pw2wannier90.x < $seedname.pw2w90 | tee $seedname.pw2w90.out
    mpirun -n 1 wannier90.x $seedname
done

mpirun python3 wannier.py
