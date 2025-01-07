#!/usr/bin/env python3

# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt

colors = ['red', 'blue', 'black', 'gray']
labels = ['$s$', '$p_{x, y}$', '$p_z$', 'other']

x, k, eps, proj = elphmod.el.read_atomic_projections(
    'work/C.save/atomic_proj.xml', order=True)

eps *= elphmod.misc.Ry

orbitals = elphmod.el.read_projwfc_out('projwfc.out')

width = 0.5 * elphmod.el.proj_sum(proj, orbitals, 's', 'p{x, y}', 'pz',
    other=True)

pwi = elphmod.bravais.read_pwi('scf.in')
mu = elphmod.el.read_Fermi_level('scf.out')

path = 'GZ'
K, X, corners = elphmod.bravais.path(path, N=100, **pwi)
X *= x[-1] / X[-1]

el = elphmod.el.Model('C')
E, order = elphmod.dispersion.dispersion(el.H, K, order=True)
E -= mu

if elphmod.MPI.comm.rank != 0:
    raise SystemExit

plt.ylabel('Electron energy (eV)')
plt.xlabel('Wave vector')
plt.xticks(X[corners], path)

for n in range(eps.shape[1]):
    fatbands = elphmod.plot.compline(x, eps[:, n], width[:, n, :])

    for fatband, color, label in zip(fatbands, colors, labels):
        plt.fill(*fatband, color=color, linewidth=0.0,
            label=None if n else label)

for n in range(el.size):
    plt.plot(X, E[:, n], 'y:', label=None if n else 'W90')

plt.legend()
plt.show()
