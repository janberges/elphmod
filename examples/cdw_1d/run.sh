#!/bin/bash

# Copyright (C) 2017-2023 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

source elphmodenv

echo 'Using normconserving pseudopotentials from PseudoDojo'
echo '[1] van Setten et al., Comput. Phys. Commun. 226, 39 (2018)'
echo '[2] Hamann, Phys. Rev. B 88, 085117 (2013)'

url=http://www.pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard
pp=C.upf
test -e $pp || (wget $url/$pp.gz && gunzip $pp)

nk=2

mpirun pw.x -nk $nk < scf.in | tee scf.out
mpirun ph.x -nk $nk < ph.in | tee ph.out

ph2epw

mpirun pw.x -nk $nk < nscf.in | tee nscf.out
mpirun -n $nk epw.x -nk $nk < epw.in | tee epw.out

mpirun python3 cdw_1d.py
