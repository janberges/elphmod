#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt
import numpy as np

comm = elphmod.MPI.comm

pwi = elphmod.bravais.read_pwi('scf.in')
mu = elphmod.el.read_Fermi_level('scf.out')

k, x, GMKG = elphmod.bravais.path('GMKG', **pwi, N=100 * np.sqrt(3) * pwi['a'])
x *= 2 * np.pi

for seedname, ref, res in ('ws_yes', 'm', 'k--'), ('ws_no', 'g', 'k:'):
    el = elphmod.el.Model('%s_hr.dat' % seedname)
    e = elphmod.dispersion.dispersion(el.H, k)

    X, E = elphmod.el.read_bands_plot('%s_band.dat' % seedname, bands=el.size)

    if comm.rank == 0:
        for n in range(el.size):
            plt.plot(X, E[:, n], ref,
                label=None if n else 'directly from W90 (%s)' % seedname)

        for n in range(el.size):
            plt.plot(x, e[:, n], res,
                label=None if n else 'via elphmod (%s)' % seedname)

if comm.rank == 0:
    plt.xlabel('wave vector')
    plt.ylabel('energy (eV)')
    plt.xticks(x[GMKG], 'GMKG')
    plt.legend()
    plt.show()
