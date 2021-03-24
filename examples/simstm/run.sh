#!/bin/bash

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

for pp in S.pbe-hgh.UPF Ta.pbe-hgh.UPF
do
    test -e $pp || wget https://www.quantum-espresso.org/upf_files/$pp
done

nk=2

mpirun pw.x -nk $nk < pw.in | tee pw.out
mpirun pp.x -nk $nk < pp.in | tee pp.out

mpirun python3 simstm.py
