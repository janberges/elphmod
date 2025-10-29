#!/usr/bin/env python3

# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt
import numpy as np

comm = elphmod.MPI.comm
info = elphmod.MPI.info

mu = -0.1665

colors = ['orange', 'skyblue', 'dodgerblue']

info('Set up Wannier Hamiltonian')

el = elphmod.el.Model('data/NbSe2')

info('Diagonalize Hamiltonian along G-M-K-G')

path = 'GMKG'
k, x, corners = elphmod.bravais.path(path, ibrav=4, N=150)

e, U, order = elphmod.dispersion.dispersion(el.H, k,
    vectors=True, order=True)

e -= mu

info('Diagonalize Hamiltonian on uniform mesh')

E = elphmod.dispersion.dispersion_full(el.H, 72) - mu

info('Calculate density of states')

w = np.linspace(E.min(), E.max(), 150)

DOS = 0

for n in range(el.size):
    DOS = DOS + elphmod.dos.hexDOS(E[:, :, n])(w)

info('Plot dispersion and density of states')

if comm.rank == 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    ax1.set_ylabel('Electron energy (eV)')
    ax1.set_xlabel('Wave vector')
    ax2.set_xlabel('Density of states (1/eV)')

    ax1.set_xticks(x[corners])
    ax1.set_xticklabels(path)

    for n in range(el.size):
        fatbands = elphmod.plot.compline(x, e[:, n],
            0.1 * (U[:, :, n] * U[:, :, n].conj()).real)

        for fatband, color in zip(fatbands, colors):
            ax1.fill(*fatband, color=color, linewidth=0.0)

    ax2.fill(DOS, w, color=colors[0], linewidth=0.0)

    plt.savefig('electrons.png')
    plt.show()
