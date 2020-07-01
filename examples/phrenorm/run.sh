#!/bin/bash

for PP in S.pbe-hgh.UPF Ta.pbe-hgh.UPF
do
    test -e $PP || wget https://www.quantum-espresso.org/upf_files/$PP
done

NK=2

mpirun pw.x -nk $NK < scf.in
mpirun ph.x -nk $NK < dfpt.in
mpirun ph.x -nk $NK < cdfpt.in
mpirun pw.x -nk $NK < nscf.in
mpirun epw.x < epw.in
