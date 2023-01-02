#!/usr/bin/env python3

# Copyright (C) 2017-2023 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt

comm = elphmod.MPI.comm

mu = -0.1665
nk = 24

a1, a2 = elphmod.bravais.translations()
b1, b2 = elphmod.bravais.reciprocals(a1, a2)

el = elphmod.el.Model('data/NbSe2')

e = elphmod.dispersion.dispersion_full(el.H, nk)[:, :, 0]

for contour in elphmod.dos.isoline(e)(mu):
    if comm.rank == 0:
        plt.plot(*zip(*[k1 * b1 + k2 * b2 for k1, k2 in contour]), 'k')

if comm.rank == 0:
    plt.plot(*zip(*elphmod.bravais.BZ()), color='k')

    plt.axis('image')
    plt.axis('off')
    plt.show()
