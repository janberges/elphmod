#!/bin/bash

for pp in S.pbe-hgh.UPF Ta.pbe-hgh.UPF
do
    test -e $pp || wget https://www.quantum-espresso.org/upf_files/$pp
done

nk=2

for method in dfpt cdfpt
do
    mpirun pw.x -nk $nk < scf.in | tee scf.out
    mpirun ph.x -nk $nk < $method.in | tee $method.out

    mpirun q2r.x < q2r.in | tee q2r.out
    ./ph2epw.sh

    mpirun pw.x -nk $nk < nscf.in | tee nscf.out
    mpirun -n 1 epw.x -nk 1 < epw.in | tee epw.out

    cp ifc $method.ifc
    cp work/TaS2.epmatwp1 $method.epmatwp1
done

mpirun python3 plot_electrons.py
mpirun python3 plot_phonons.py
