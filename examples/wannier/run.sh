#!/bin/bash

# Copyright (C) 2017-2023 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

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
