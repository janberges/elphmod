#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod

AFMhot = elphmod.plot.colormap( # Gnuplot
    (0.00, elphmod.plot.Color(0, 0, 0)),
    (0.25, elphmod.plot.Color(128, 0, 0)),
    (0.50, elphmod.plot.Color(255, 128, 0)),
    (0.75, elphmod.plot.Color(255, 255, 128)),
    (1.00, elphmod.plot.Color(255, 255, 255)),
    )

a, X, r, data = elphmod.misc.read_cube('stm.cube')

plot = elphmod.plot.plot(data[:, :, data.shape[2] // 4], angle=120)

image = elphmod.plot.color(plot, AFMhot)

if elphmod.MPI.comm.rank == 0:
    elphmod.plot.save('simstm.png', image)
