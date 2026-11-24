#!/usr/bin/env python3

# Copyright (C) 2017-2026 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt
import numpy as np

colors = ['red', 'green', 'magenta', 'orange', 'blue']
labels = ['$d_{z^2}$', '$d_{x z}$', '$d_{y z}$', '$d_{x^2 - y^2}$', '$d_{x y}$']

for number, label in enumerate(['', '_ncl', '_soc']):
    x, k, eps, proj = elphmod.el.read_atomic_projections(
        'work/TaS2%s.save/atomic_proj.xml' % label.upper(), squared=False)

    eps *= elphmod.misc.Ry

    orbitals = elphmod.el.read_projwfc_out('projwfc%s.out' % label)

    if label == '_soc':
        d = [i for i, orbital in enumerate(orbitals) if 'd' in orbital]

        proj[:, :, d] @= elphmod.el.clebsh_gordan(2).T
        proj[:, :, d] @= elphmod.el.complex_to_real_orbitals(2).conj().T

        for i, orbital in zip(d, ['Ta-d%s%s' % (m, s)
                for m in ['xy', 'yz', 'z2', 'xz', 'x2-y2'] for s in 'du']):
            orbitals[i] = orbital

    proj = abs(proj) ** 2

    width = 0.2 * elphmod.el.proj_sum(proj, orbitals,
        'dz2{u,d,}', 'dxz{u,d,}', 'dyz{u,d,}', 'dx2-y2{u,d,}', 'dxy{u,d,}')

    pwi = elphmod.bravais.read_pwi('scf%s.in' % label)

    path = 'GMKG'
    K, X, corners = elphmod.bravais.path(path, N=100, **pwi)
    X *= x[-1] / X[-1]

    if elphmod.MPI.comm.rank != 0:
        continue

    plt.ylabel('Electron energy (eV)')
    plt.xlabel('Wave vector')
    plt.xticks(X[corners], path)

    n0 = np.argmax(np.any(eps[:, :] > 0, axis=0))

    for n in range(n0, eps.shape[1]):
        fatbands = elphmod.plot.compline(x, eps[:, n], width[:, n])

        for fatband, color, label in zip(fatbands, colors, labels):
            plt.fill(*fatband, color=color, linewidth=0.0,
                label=label if n == n0 else None)

    plt.legend()
    plt.savefig('projwfc_tas2_%d.png' % number)
    plt.show()
