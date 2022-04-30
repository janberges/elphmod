#!/usr/bin/env python3

import elphmod
import matplotlib.pyplot as plt

colors = ['red', 'blue', 'green', 'gray']
labels = ['$s$', '$p$', '$d$', 'other']

x, k, eps, proj = elphmod.el.read_atomic_projections(
    'work/polonium.save/atomic_proj.xml', order=False, other=True)

eps *= elphmod.misc.Ry

orbitals = elphmod.el.read_projwfc_out('projwfc.out')

width = 0.5 * elphmod.el.proj_sum(proj, orbitals, 's', 'p', 'd', 'x')

if elphmod.MPI.comm.rank != 0:
    raise SystemExit

K, X, GXRMG = elphmod.bravais.path('GXRMG', ibrav=1)
X *= x[-1] / X[-1]

plt.ylabel('Electron energy (eV)')
plt.xlabel('Wave vector')
plt.xticks(X[GXRMG], 'GXRMG')

for n in range(eps.shape[1]):
    fatbands = elphmod.plot.compline(x, eps[:, n], width[:, n, :])

    for fatband, color, label in zip(fatbands, colors, labels):
        plt.fill(*fatband, color=color, linewidth=0.0,
            label=None if n else label)

plt.legend()
plt.show()
