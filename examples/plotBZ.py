#/usr/bin/env python

import elphmod

import os

comm = elphmod.MPI.comm
info = elphmod.MPI.info

Ry2eV = 13.605693009

info("Read and fix force constants and set up dynamical matrix..")

model = elphmod.phonons.model('data/NbSe2-DFPT-LR.ifc', apply_asr=True)

D = elphmod.phonons.dynamical_matrix(*model)

info("Calculate dispersion on whole Brillouin zone..")

nq = 48

w2 = elphmod.dispersion.dispersion_full(D, nq, order=False, broadcast=True)

info("Plot dispersion on Brillouin zone..")

w = elphmod.phonons.sgnsqrt(w2) * Ry2eV * 1e3

elphmod.plot.plot_pie_with_TeX(
    'plotBZ.tex', [w[:, :, nu] for nu in range(6)],
    ticks=range(-10, 30, 10), title=r'Phonon frequency', unit='meV',
    form=lambda x: r'$%g\,\mathrm{i}$' % abs(x) if x < 0 else '$%g$' % x)

if comm.rank == 0:
    os.system('mkdir -p plotBZ')
    os.chdir('plotBZ')
    os.system('pdflatex plotBZ > /dev/null')
    os.chdir('..')
