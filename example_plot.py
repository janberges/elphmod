#/usr/bin/env python

import os
import phonons
import plot

from mpi4py import MPI
comm = MPI.COMM_WORLD

Ry2eV = 13.605693009

if comm.rank == 0:
    print("Read and fix force constants and set up dynamical matrix..")

    model = phonons.read_flfrc('data/NbSe2-DFPT-LR.ifc')

    phonons.asr(model[0])
else:
    model = None

model = comm.bcast(model)

D = phonons.dynamical_matrix(comm, *model)

if comm.rank == 0:
    print("Calculate dispersion on whole Brillouin zone..")

nq = 48

w = phonons.dispersion(comm, D, nq, order=False) * Ry2eV

if comm.rank == 0:
    print("Plot dispersion on Brillouin zone..")

    os.system('mkdir -p example_plot')
    os.chdir('example_plot')

    plot.plot_pie_with_TeX('BZ.tex', [w[:, :, nu] for nu in range(6)])

    os.system('pdflatex BZ > /dev/null')
    os.chdir('..')
