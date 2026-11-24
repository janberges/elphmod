#!/bin/bash

# Copyright (C) 2017-2026 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

eval `elphmodenv`

echo 'Using normconserving pseudopotentials from PseudoDojo'
echo '[1] van Setten et al., Comput. Phys. Commun. 226, 39 (2018)'
echo '[2] Hamann, Phys. Rev. B 88, 085117 (2013)'

for typ in sr fr
do
    url=http://www.pseudo-dojo.org/pseudos/nc-$typ-04_pbe_standard # [1, 2]

    for x in Ta S
    do
        pp=$x.upf
        pp_typ=$x-$typ.upf
        test -e $pp_typ || (wget $url/$pp.gz && gunzip $pp && mv $pp $pp_typ)
    done
done

: ${NP:=2}
: ${NK:=2}

for label in '' _ncl _soc
do
    mpirun -n $NP pw.x -nk $NK < scf$label.in | tee scf$label.out
    mpirun -n $NP pw.x -nk $NK < bands$label.in | tee bands$label.out
    mpirun -n $NP projwfc.x -nk $NK < projwfc$label.in | tee projwfc$label.out
done

mpirun -n $NP python3 projwfc_tas2.py
