#!/usr/bin/env python3

# Copyright (C) 2017-2023 elphmod Developers
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

el = elphmod.el.Model('data/NbSe2')
El = el.supercell(*N)

path = 'GMKG'
k, x, corners = elphmod.bravais.path(path, ibrav=4, N=150)
K = np.dot(np.dot(k, b), A.T)

e, u = elphmod.dispersion.dispersion(el.H, k, vectors=True)
E, U = elphmod.dispersion.dispersion(El.H, K, vectors=True)

w = np.ones(e.shape)
W = elphmod.dispersion.unfolding_weights(k, El.cells, u, U)

linewidth = 0.1

if elphmod.MPI.comm.rank == 0:
    for n in range(e.shape[1]):
        fatband, = elphmod.plot.compline(x, e[:, n], linewidth * w[:, n])

        plt.fill(*fatband, linewidth=0.0, color='skyblue')

    for n in range(E.shape[1]):
        fatband, = elphmod.plot.compline(x, E[:, n], linewidth * W[:, n])

        plt.fill(*fatband, linewidth=0.0, color='dodgerblue')

    plt.ylabel('Electron energy (eV)')
    plt.xlabel('Wave vector')
    plt.xticks(x[corners], path)
    plt.show()
