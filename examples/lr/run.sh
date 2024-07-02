#!/bin/bash

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

eval `elphmodenv`

echo 'Using normconserving pseudopotentials from PseudoDojo'
echo '[1] van Setten et al., Comput. Phys. Commun. 226, 39 (2018)'
echo '[2] Hamann, Phys. Rev. B 88, 085117 (2013)'

url=http://www.pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard

for pp in Mo.upf S.upf
do
    test -e $pp || (wget $url/$pp.gz && gunzip $pp)
done

nk=2

python3 lr.py --prepare-q

mpirun pw.x -nk $nk < pw.in | tee pw.out
mpirun ph.x -nk $nk < ph.in | tee ph.out

for lr in '3d' 'gaussian'
do
    mpirun q2r.x < q2r_$lr.in | tee q2r_$lr.out
    mpirun matdyn.x < matdyn_$lr.in | tee matdyn_$lr.out
done

ph2epw

mpirun pw.x -nk $nk < nscf.in | tee nscf.out

for lr in 'no_lr' '3d' 'gaussian' 'dipole_sp' 'quadrupole'
do
    test $lr = 'quadrupole' && mv _quadrupole.fmt quadrupole.fmt

    mpirun -n $nk epw.x -nk $nk < epw_$lr.in | tee epw_$lr.out

    test $lr = 'quadrupole' && mv quadrupole.fmt _quadrupole.fmt

    mv work/MoS2.epmatwp $lr.epmatwp
done

mpirun python3 lr.py
