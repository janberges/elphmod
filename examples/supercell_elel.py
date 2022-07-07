#!/usr/bin/env python3

# Copyright (C) 2017-2022 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt
import numpy as np

N = [ # 3 x 3 (60 degrees instead of 120 degrees)
    [3, 0, 0],
    [3, 3, 0],
    [0, 0, 1],
    ]

a = elphmod.bravais.primitives(ibrav=4)
b = elphmod.bravais.reciprocals(*a)
A = np.dot(N, a)

elel = elphmod.elel.Model('data/U.ijkl', nq=2, no=1)
ElEl = elel.supercell(*N)

path = 'GMKG'
q, x, corners = elphmod.bravais.path(path, ibrav=4, N=150)
Q = np.dot(np.dot(q, b), A.T)

v, u = elphmod.dispersion.dispersion(elel.W, q, vectors=True)
V, U = elphmod.dispersion.dispersion(ElEl.W, Q, vectors=True)

w = np.ones(v.shape)
W = elphmod.dispersion.unfolding_weights(q, ElEl.cells, u, U)

linewidth = 0.05

if elphmod.MPI.comm.rank == 0:
    for n in range(v.shape[1]):
        fatband, = elphmod.plot.compline(x, v[:, n], linewidth * w[:, n])

        plt.fill(*fatband, linewidth=0.0, color='violet')

    for n in range(V.shape[1]):
        fatband, = elphmod.plot.compline(x, V[:, n], linewidth * W[:, n])

        plt.fill(*fatband, linewidth=0.0, color='purple')

    plt.ylabel('Coulomb interaction (eV)')
    plt.xlabel('Wave vector')
    plt.xticks(x[corners], path)
    plt.show()
