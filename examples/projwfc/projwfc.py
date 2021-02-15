#!/usr/bin/env python3

import elphmod
import matplotlib.pyplot as plt
import numpy as np

colors = ['magenta', 'cyan', 'red', 'blue', 'green', 'black']
labels = ['S-$s$', 'S-$p$', 'Ta-$s$', 'Ta-$p$', 'Ta-$d_{x z, y z}$',
    'Ta-$d_{z^2, x^2 - y^2, x y}$']

x, k, eps, proj = elphmod.el.read_atomic_projections(
    'work/TaS2.save/atomic_proj.xml', order=True)

eps *= elphmod.misc.Ry

orbitals = elphmod.el.read_projwfc_out('projwfc.out')

width = 0.5 * elphmod.el.proj_sum(proj, orbitals, 'S-s', 'S-p',
    'Ta-s', 'Ta-p', 'Ta-d{xz, yz}', 'Ta-d{z2, x2-y2, xy}')

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
