#!/bin/bash

# Copyright (C) 2017-2022 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

url=https://pseudopotentials.quantum-espresso.org/upf_files

for pp in S.pbe-hgh.UPF Ta.pbe-hgh.UPF
do
    test -e $pp || wget $url/$pp
done

nk=2

mpirun pw.x -nk $nk < pw.in | tee pw.out
mpirun ph.x -nk $nk < ph.in | tee ph.out
mpirun q2r.x < q2r.in | tee q2r.out

mpirun python3 bare.py
