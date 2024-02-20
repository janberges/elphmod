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

mpirun pw.x -nk $nk < pw.in | tee pw.out
mpirun ph.x -nk $nk < ph.in | tee ph.out

mpirun python3 bare.py
