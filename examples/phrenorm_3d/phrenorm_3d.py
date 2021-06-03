#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import copy
import elphmod
import matplotlib.pyplot as plt
import numpy as np

comm = elphmod.MPI.comm
info = elphmod.MPI.info

kT = 0.1 * elphmod.misc.Ry
f = elphmod.occupations.gauss

nk = 4
nq = 2

info('Prepare wave vectors')

k = 2 * np.pi * np.array([[[(k1, k2, k3)
    for k3 in range(nk)]
    for k2 in range(nk)]
    for k1 in range(nk)], dtype=float) / nk

q = 2 * np.pi * np.array([[[(q1, q2, q3)
    for q3 in range(nq)]
    for q2 in range(nq)]
    for q1 in range(nq)], dtype=float) / nq

q_flat = np.reshape(q, (-1, 3))

path, x, GXRMG = elphmod.bravais.path('GXRMG', ibrav=1)

info('Prepare electrons')

el = elphmod.el.Model('polonium_hr.dat')
mu = elphmod.el.read_Fermi_level('scf.out')

e, U = elphmod.dispersion.dispersion(el.H, k, vectors=True)
e -= mu

info('Prepare phonons')

ph = dict()

for method in 'cdfpt', 'dfpt':
    ph[method] = elphmod.ph.Model('%s.ifc' % method, apply_asr=True)

info('Prepare electron-phonon coupling')

g = dict()

for method in sorted(ph):
    elph = elphmod.elph.Model('%s.epmatwp' % method, 'wigner.dat',
        el, ph['cdfpt'])

    g[method] = elph.sample(q=q_flat, U=U, u=None, broadcast=False)

if comm.rank == 0:
    g2 = np.einsum('qixyzmn,qjxyzmn->qijxyzmn', g['cdfpt'].conj(), g['dfpt'])
    g2 *= elphmod.misc.Ry ** 3

    g2 += np.einsum('qijxyzmn->qjixyzmn', g2.conj())
    g2 /= 2

else:
    g2 = np.empty((len(q_flat), ph['cdfpt'].size, ph['cdfpt'].size,
        nk, nk, nk, el.size, el.size), dtype=complex)

comm.Bcast(g2)

info('Calculate phonon self-energy')

Pi = elphmod.diagrams.phonon_self_energy(q_flat, e, g2, kT=kT, occupations=f)
Pi = np.reshape(Pi, (nq, nq, nq, ph['cdfpt'].size, ph['cdfpt'].size))
Pi /= elphmod.misc.Ry ** 2

info('Renormalize phonons')

D = elphmod.dispersion.sample(ph['cdfpt'].D, q)

ph['cdfpt+pi'] = copy.copy(ph['cdfpt'])

elphmod.ph.q2r(ph['cdfpt+pi'], D_full=D + Pi, apply_asr=True)

info('Plot electrons')

e = elphmod.dispersion.dispersion(el.H, path)
e -= mu

if comm.rank == 0:
    for n in range(el.size):
        plt.plot(x, e[:, n], 'b')

    plt.ylabel(r'$\epsilon$ (eV)')
    plt.xlabel(r'$k$')
    plt.xticks(x[GXRMG], 'GXRMG')
    plt.show()

info('Plot cDFPT, DFPT and renormalized phonons')

for method, label, style in [('dfpt', 'DFPT', 'r'), ('cdfpt', 'cDFPT', 'g'),
        ('cdfpt+pi', r'cDFPT+$\Pi$', 'b:')]:

    w2 = elphmod.dispersion.dispersion(ph[method].D, path)

    w = elphmod.ph.sgnsqrt(w2) * elphmod.misc.Ry * 1e3

    if comm.rank == 0:
        for nu in range(ph[method].size):
            plt.plot(x, w[:, nu], style, label=None if nu else label)

if comm.rank == 0:
    plt.ylabel(r'$\omega$ (meV)')
    plt.xlabel(r'$q$')
    plt.xticks(x[GXRMG], 'GXRMG')
    plt.legend()
    plt.show()
