#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt
import numpy as np

nu = 8 # ionic displacement
N = [2, 2, 1]

a = elphmod.bravais.primitives(ibrav=4)
b = elphmod.bravais.reciprocals(*a)
A = np.dot(np.diag(N), a)

el = elphmod.el.Model('TaS2')
ph = elphmod.ph.Model('dfpt.ifc', apply_asr=True)
elph = elphmod.elph.Model('dfpt.epmatwp', 'wigner.dat', el, ph,
    divide_mass=False)
ElPh = elph.supercell_general(*N)

k, x, GMKG = elphmod.bravais.path('GMKG', ibrav=4, N=300)
K = np.dot(np.dot(k, b), A.T)

g = np.empty((len(k), elph.el.size), dtype=complex)
G = np.empty((len(K), ElPh.el.size), dtype=complex)

for ik, ((k1, k2, k3), (K1, K2, K3)) in enumerate(zip(k, K)):
    g[ik] = np.linalg.eigvals(elph.g(k1=k1, k2=k3, k3=k3)[nu])
    G[ik] = np.linalg.eigvals(ElPh.g(k1=K1, k2=K3, k3=K3)[nu::ph.size].sum(0))
    g[ik].sort()
    G[ik].sort()

g *= elphmod.misc.Ry / elphmod.misc.a0
G *= elphmod.misc.Ry / elphmod.misc.a0

if elphmod.MPI.comm.rank == 0:
    plt.xticks(x[GMKG], 'GMKG')
    plt.ylabel(r'$\partial V / \partial z_{\mathrm{S}}$ ($\mathrm{eV/\AA}$)')
    plt.plot(x, g.real, 'k')
    plt.plot(x, G.real, 'y:')
    plt.show()
