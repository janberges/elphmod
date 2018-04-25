#/usr/bin/env python

import elphmod

import os

from mpi4py import MPI
comm = MPI.COMM_WORLD

Ry2eV = 13.605693009

if comm.rank == 0:
    print("Read and fix force constants and set up dynamical matrix..")

    model = elphmod.phonons.read_flfrc('data/NbSe2-DFPT-LR.ifc')

    elphmod.phonons.asr(model[0])
else:
    model = None

model = comm.bcast(model)

D = elphmod.phonons.dynamical_matrix(comm, *model)

if comm.rank == 0:
    print("Calculate dispersion on whole Brillouin zone..")

nq = 48

w = elphmod.phonons.dispersion(comm, D, nq, order=False) * Ry2eV * 1e3

if comm.rank == 0:
    print("Plot dispersion on Brillouin zone..")

    os.system('mkdir -p example_plot')
    os.chdir('example_plot')

    elphmod.plot.plot_pie_with_TeX('BZ.tex', [w[:, :, nu] for nu in range(6)],
        ticks=range(-10, 30, 10), title=r'Phonon frequency', unit='meV',
        form=lambda x: r'$%g\,\mathrm{i}$' % abs(x) if x < 0 else '$%g$' % x)

    os.system('pdflatex BZ > /dev/null')
    os.chdir('..')
