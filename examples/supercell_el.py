#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt
import numpy as np

N1 = 3
N2 = 3
N3 = 1

el = elphmod.el.Model('data/NbSe2')
El = el.supercell(N1, N2, N3)

k, x, GMKG = elphmod.bravais.path('GMKG', ibrav=4, N=150)
K = np.array([N1, N2, N3]) * k

e, u = elphmod.dispersion.dispersion(el.H, k, vectors=True)
E, U = elphmod.dispersion.dispersion(El.H, K, vectors=True)

R = []
blocks0 = []
blocks = []

for n1 in range(N1):
    for n2 in range(N2):
        for n3 in range(N3):
            R.append((n1, n2, n3))
            blocks0.append(slice(el.size))
            offset = (n1 * N2 * N3 + n2 * N3 + n3) * el.size
            blocks.append(slice(offset, offset + el.size))

w = np.ones(e.shape)
W = elphmod.dispersion.unfolding_weights(k, R, u, U, blocks0, blocks)

linewidth = 0.1

if elphmod.MPI.comm.rank == 0:
    for n in range(e.shape[1]):
        fatband, = elphmod.plot.compline(x, e[:, n], linewidth * w[:, n])

        plt.fill(*fatband, linewidth=0.0, color='skyblue')

    for n in range(E.shape[1]):
        fatband, = elphmod.plot.compline(x, E[:, n], linewidth * W[:, n])

        plt.fill(*fatband, linewidth=0.0, color='dodgerblue')

    plt.ylabel('electron energy (eV)')
    plt.xlabel('wave vector')
    plt.xticks(x[GMKG], 'GMKG')
    plt.show()
