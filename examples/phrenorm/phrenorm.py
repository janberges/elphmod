#!/usr/bin/env python3

# Copyright (C) 2017-2022 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import copy
import elphmod
import matplotlib.pyplot as plt
import numpy as np

comm = elphmod.MPI.comm
info = elphmod.MPI.info

PW = elphmod.bravais.read_pwi('scf.in')
PH = elphmod.bravais.read_ph('dfpt.in')

kT = PW['degauss'] * elphmod.misc.Ry
f = elphmod.occupations.smearing(PW['smearing'])

nk = PW['k_points'][0]
nq = PH['nq1']

nel = 1
nph = 9

info('Prepare wave vectors')

q = sorted(elphmod.bravais.irreducibles(nq))
q = 2 * np.pi * np.array(q, dtype=float) / nq

k, x, GMKG = elphmod.bravais.GMKG(100, corner_indices=True)

info('Prepare electrons')

el = elphmod.el.Model('TaS2')
mu = elphmod.el.read_Fermi_level('scf.out')

e, U = elphmod.dispersion.dispersion_full_nosym(el.H, nk, vectors=True)

e = e[..., :nel] - mu
U = U[..., :nel]

info('Prepare phonons')

ph = dict()

for method in 'cdfpt', 'dfpt':
    ph[method] = elphmod.ph.Model('%s.ifc' % method, apply_asr_simple=True)

info('Prepare electron-phonon coupling')

g = dict()

for method in sorted(ph):
    elph = elphmod.elph.Model('%s.epmatwp' % method, 'wigner.dat',
        el, ph[method])

    g[method] = elph.sample(q, U=U, u=None) * elphmod.misc.Ry ** 1.5

info('Calculate phonon self-energy')

Pi = elphmod.diagrams.phonon_self_energy(q, e, g=g['cdfpt'], G=g['dfpt'],
    kT=kT, occupations=f) / elphmod.misc.Ry ** 2

info('Renormalize phonons')

D = elphmod.dispersion.sample(ph['cdfpt'].D, q)

ph['pp'] = copy.copy(ph['dfpt'])

elphmod.ph.q2r(ph['pp'], D + Pi, q, nq, apply_asr_simple=True)

info('Plot electrons')

e, U, order = elphmod.dispersion.dispersion(el.H, k, vectors=True, order=True)
e -= mu

if comm.rank == 0:
    proj = 0.1 * abs(U) ** 2

    for n in range(el.size):
        fatbands = elphmod.plot.compline(x, e[:, n], proj[:, :, n])

        for fatband, color in zip(fatbands, 'rgb'):
            plt.fill(*fatband, color=color, linewidth=0.0)

    plt.ylabel('Electron energy (eV)')
    plt.xlabel('Wave vector')
    plt.xticks(x[GMKG], 'GMKG')
    plt.show()

info('Plot cDFPT, DFPT and renormalized phonons')

for method in sorted(ph):
    w2, u, order = elphmod.dispersion.dispersion(ph[method].D, k,
        vectors=True, order=True)

    w = elphmod.ph.sgnsqrt(w2) * elphmod.misc.Ry * 1e3

    if comm.rank == 0:
        proj = elphmod.ph.polarization(u, k)

        for nu in range(ph[method].size):
            fatbands = elphmod.plot.compline(x, w[:, nu], proj[:, nu])

            for fatband, color in zip(fatbands, 'ymk'):
                plt.fill(*fatband, color=color, linewidth=0.0)

        plt.ylabel('Phonon energy (meV)')
        plt.xlabel('Wave vector')
        plt.xticks(x[GMKG], 'GMKG')
        plt.show()
