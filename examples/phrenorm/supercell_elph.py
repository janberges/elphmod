#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt
import numpy as np

nu = 8 # displacement direction

N1 = 2
N2 = 2
N3 = 1

el = elphmod.el.Model('TaS2')
ph = elphmod.ph.Model('dfpt.ifc', apply_asr=True)
elph = elphmod.elph.Model('dfpt.epmatwp', 'wigner.dat', el, ph,
    divide_mass=False)
elph.data *= elphmod.misc.Ry / elphmod.misc.a0

ElPh = elph.supercell(N1, N2, N3)

k, x, GMKG = elphmod.bravais.path('GMKG', ibrav=4, N=300)
K = np.array([N1, N2, N3]) * k

I = elphmod.MPI.comm.Split(elphmod.MPI.comm.rank)

def g(k1=0.0, k2=0.0, k3=0.0):
    return elph.g(q1=0.0, q2=0.0, q3=0.0, k1=k1, k2=k2, k3=k3,
        comm=I)[nu]

def G(K1=0.0, K2=0.0, K3=0.0):
    return ElPh.g(q1=0.0, q2=0.0, q3=0.0, k1=K1, k2=K2, k3=K3,
        comm=I)[nu::ph.size].sum(axis=0)

g, u = elphmod.dispersion.dispersion(g, k, vectors=True)
G, U = elphmod.dispersion.dispersion(G, K, vectors=True)

R = []
blocks = []

for n1 in range(N1):
    for n2 in range(N2):
        for n3 in range(N3):
            R.append((n1, n2, n3))
            offset = (n1 * N2 * N3 + n2 * N3 + n3) * el.size
            blocks.append(slice(offset, offset + el.size))

w = np.ones(g.shape)
W = elphmod.dispersion.unfolding_weights(k, R, u, U, blocks=blocks)

linewidth = 0.1

if elphmod.MPI.comm.rank == 0:
    for n in range(g.shape[1]):
        fatband, = elphmod.plot.compline(x, g[:, n], linewidth * w[:, n])

        plt.fill(*fatband, linewidth=0.0, color='skyblue')

    for n in range(G.shape[1]):
        fatband, = elphmod.plot.compline(x, G[:, n], linewidth * W[:, n])

        plt.fill(*fatband, linewidth=0.0, color='dodgerblue')

    plt.ylabel('electron-phonon coupling ($\mathrm{eV/\AA}$)')
    plt.xlabel('wave vector')
    plt.xticks(x[GMKG], 'GMKG')
    plt.show()
