#!/usr/bin/env python3

# Copyright (C) 2017-2022 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt
import scipy.optimize

comm = elphmod.MPI.comm

colors = ['skyblue', 'dodgerblue', 'orange']

ph = elphmod.ph.Model('dyn')

def cost(L):
    ph.L, = L
    ph.update_short_range()
    return ph.sum_force_constants()

scipy.optimize.minimize(cost, [10.0])

q, x, GMKG = elphmod.bravais.GMKG(300, corner_indices=True)

w2, u, order = elphmod.dispersion.dispersion(ph.D, q,
    vectors=True, order=True)

if comm.rank == 0:
    w = elphmod.ph.sgnsqrt(w2) * elphmod.misc.Ry * 1e3

    pol = elphmod.ph.polarization(u, q)

    for nu in range(ph.size):
        fatbands = elphmod.plot.compline(x, w[:, nu], 2 * pol[:, nu])

        for fatband, color in zip(fatbands, colors):
            plt.fill(*fatband, color=color, linewidth=0.0)

    plt.ylabel('Phonon energy (meV)')
    plt.xlabel('Wave vector')
    plt.xticks(x[GMKG], 'GMKG')
    plt.show()
