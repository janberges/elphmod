#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt
import numpy as np

N1 = 3
N2 = 3
N3 = 1

ph = elphmod.ph.Model('data/NbSe2_cDFPT.ifc', apply_asr_simple=True)
Ph = ph.supercell(N1, N2, N3)

q, x, GMKG = elphmod.bravais.path('GMKG', ibrav=4, N=150)
Q = np.array([N1, N2, N3]) * q

w2, u = elphmod.dispersion.dispersion(ph.D, q, vectors=True)
W2, U = elphmod.dispersion.dispersion(Ph.D, Q, vectors=True)

R = []
blocks0 = []
blocks = []

for n1 in range(N1):
    for n2 in range(N2):
        for n3 in range(N3):
            R.append((n1, n2, n3))
            blocks0.append(slice(ph.size))
            offset = (n1 * N2 * N3 + n2 * N3 + n3) * ph.size
            blocks.append(slice(offset, offset + ph.size))

w = np.ones(w2.shape)
W = elphmod.dispersion.unfolding_weights(q, R, u, U, blocks0, blocks, sgn=+1)

linewidth = 1.0

if elphmod.MPI.comm.rank == 0:
    for nu in range(w2.shape[1]):
        fatband, = elphmod.plot.compline(x, elphmod.ph.sgnsqrt(w2[:, nu])
            * elphmod.misc.Ry * 1e3, linewidth * w[:, nu])

        plt.fill(*fatband, linewidth=0.0, color='red')

    for nu in range(W2.shape[1]):
        fatband, = elphmod.plot.compline(x, elphmod.ph.sgnsqrt(W2[:, nu])
            * elphmod.misc.Ry * 1e3, linewidth * W[:, nu])

        plt.fill(*fatband, linewidth=0.0, color='firebrick')

    plt.ylabel('phonon energy (meV)')
    plt.xlabel('wave vector')
    plt.xticks(x[GMKG], 'GMKG')
    plt.show()
