#!/usr/bin/env python3

# Copyright (C) 2017-2023 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt

r0, a, X, tau, data = elphmod.misc.read_cube('stm.cube')

plot = elphmod.plot.plot(data[:, :, data.shape[2] // 4], angle=120)

if elphmod.MPI.comm.rank == 0:
    plt.imshow(plot, cmap='afmhot')
    plt.axis('off')
    plt.show()
