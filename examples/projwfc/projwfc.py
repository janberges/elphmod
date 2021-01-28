#!/usr/bin/env python3

import elphmod
import matplotlib.pyplot as plt
import numpy as np

colors = ['magenta', 'cyan', 'red', 'blue', 'green']
labels = ['S-$s$', 'S-$p$', 'Ta-$s$', 'Ta-$p$', 'Ta-$d$']

x, k, eps, proj = elphmod.el.read_atomic_projections(
    'work/TaS2.save/atomic_proj.xml', order=True)

if elphmod.MPI.comm.rank != 0:
    raise SystemExit

K, X, GMKG = elphmod.bravais.GMKG(6, mesh=True, corner_indices=True)
X *= x[-1] / X[-1]

width = np.empty(proj.shape[:2] + (len(labels),))

width[..., 0] = proj[..., [0, 14]].sum(axis=2)
width[..., 1] = proj[..., [1, 2, 3, 15, 16, 17]].sum(axis=2)
width[..., 2] = proj[..., [4, 13]].sum(axis=2)
width[..., 3] = proj[..., [5, 6, 7]].sum(axis=2)
width[..., 4] = proj[..., [8, 9, 10, 11, 12]].sum(axis=2)

width *= 0.05

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
