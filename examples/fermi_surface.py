#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod

mu = -0.1665
kT = 0.025
nk = 72
points = 200

cmap = elphmod.plot.colormap(
    (0, elphmod.plot.Color(0.0, 1, 255, 'PSV')),
    (1, elphmod.plot.Color(5.5, 1, 255, 'PSV')),
    )

el = elphmod.el.Model('data/NbSe2')

e = elphmod.dispersion.dispersion_full(el.H, nk)[:, :, 0] - mu

e = elphmod.plot.plot(e, kxmin=0.0, kxmax=1.5, kymin=0.0, kymax=1.0,
    resolution=points)

delta = elphmod.occupations.fermi_dirac.delta(e / kT)

image = elphmod.plot.color(delta, cmap)

if elphmod.MPI.comm.rank == 0:
    elphmod.plot.save('fermi_surface.png', image)
