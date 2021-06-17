#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt

colors = ['red', 'blue', 'black']
labels = ['$s$', '$p_{x, y}$', '$p_z$']

x, k, eps, proj = elphmod.el.read_atomic_projections(
    'work/graphene.save/atomic_proj.xml', order=True)

eps *= elphmod.misc.Ry

orbitals = elphmod.el.read_projwfc_out('projwfc.out')

width = 0.5 * elphmod.el.proj_sum(proj, orbitals, 's', 'p{x, y}', 'pz')

if elphmod.MPI.comm.rank != 0:
    raise SystemExit

K, X, GMKG = elphmod.bravais.GMKG(6, mesh=True, corner_indices=True)
X *= x[-1] / X[-1]

plt.xticks(X[GMKG], 'GMKG')
plt.xlabel('wave vector')
plt.ylabel('energy (eV)')

for n in range(eps.shape[1]):
    fatbands = elphmod.plot.compline(x, eps[:, n], width[:, n, :])

    for fatband, color, label in zip(fatbands, colors, labels):
        plt.fill(*fatband, color=color, linewidth=0.0,
            label=None if n else label)

plt.legend()
plt.show()
