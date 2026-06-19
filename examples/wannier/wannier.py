#!/usr/bin/env python3

# Copyright (C) 2017-2026 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt
import numpy as np

comm = elphmod.MPI.comm

pwi = elphmod.bravais.read_pwi('scf.in')

path = 'GMKG'
k, x, corners = elphmod.bravais.path(path, **pwi, N=100 * np.sqrt(3) * pwi['a'])
x *= 2 * np.pi

for seedname, ref, res in ('ws_yes', 'm', 'k--'), ('ws_no', 'g', 'k:'):
    el = elphmod.el.Model(seedname)
    e = elphmod.dispersion.dispersion(el.H, k)

    X, E = elphmod.el.read_bands_plot('%s_band.dat' % seedname, bands=el.size)

    if comm.rank == 0:
        for n in range(el.size):
            plt.plot(X, E[:, n], ref, linewidth=2,
                label=None if n else 'directly from W90 (%s)' % seedname)

        for n in range(el.size):
            plt.plot(x, e[:, n], res, linewidth=2,
                label=None if n else 'via elphmod (%s)' % seedname)

k_mesh = elphmod.bravais.mesh(*pwi['k_points'][:3])
H_mesh = elphmod.dispersion.sample(el.H, k_mesh)

elphmod.el.k2r(el, H_mesh)

e = elphmod.dispersion.dispersion(el.H, k)

if comm.rank == 0:
    for n in range(el.size):
        plt.plot(x, e[:, n], 'y-', linewidth=1,
            label=None if n else 'via elphmod (k2r)')

    plt.ylabel('Electron energy (eV)')
    plt.xlabel('Wave vector')
    plt.xticks(x[corners], path)
    plt.legend()
    plt.savefig('wannier.png')
    plt.show()
