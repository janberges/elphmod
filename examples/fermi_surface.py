#!/usr/bin/env python3

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt

mu = -0.1665
kT = 0.025
nk = 72
points = 200

el = elphmod.el.Model('data/NbSe2')

e = elphmod.dispersion.dispersion_full(el.H, nk)[:, :, 0] - mu

e = elphmod.plot.plot(e, kxmin=0.0, kxmax=1.5, kymin=0.0, kymax=1.0,
    resolution=points)

delta = elphmod.occupations.fermi_dirac.delta(e / kT)

if elphmod.MPI.comm.rank == 0:
    plt.imshow(delta, cmap='magma')
    plt.axis('off')
    plt.show()
