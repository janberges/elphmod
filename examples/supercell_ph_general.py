#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt
import numpy as np

N = [
    [ 4, 1, 0],
    [-1, 3, 0],
    [ 0, 0, 1],
    ]

a = elphmod.bravais.primitives(ibrav=4)
b = elphmod.bravais.reciprocals(*a)
A = np.dot(N, a)

ph = elphmod.ph.Model('data/NbSe2_cDFPT.ifc', apply_asr_simple=True)
Ph = ph.supercell_general(*N)

q, x, GMKG = elphmod.bravais.path('GMKG', ibrav=4, N=150)
Q = np.dot(np.dot(q, b), A.T)

w2, u = elphmod.dispersion.dispersion(ph.D, q, vectors=True)
W2, U = elphmod.dispersion.dispersion(Ph.D, Q, vectors=True)

w = np.ones(w2.shape)
W = elphmod.dispersion.unfolding_weights(q, Ph.cells, u, U, sgn=+1)

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