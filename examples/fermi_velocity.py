#!/usr/bin/env python3

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import numpy as np
import matplotlib.pyplot as plt

comm = elphmod.MPI.comm
info = elphmod.MPI.info

mu = -0.1665
nk = 120
points = 500

el = elphmod.el.Model('data/NbSe2')

e = elphmod.dispersion.dispersion_full(el.H, nk)[:, :, 0] - mu

kxmax, kymax, kx, ky, e = elphmod.plot.toBZ(e,
    points=points, return_k=True, outside=np.nan)

dedky, dedkx = np.gradient(e, ky, kx)
dedk = np.sqrt(dedkx ** 2 + dedky ** 2)

if comm.rank == 0:
    plt.imshow(dedk, cmap='Greys')
    plt.axis('off')
    plt.show()

info('Min./max./mean number of k-points for meV resolution:')

FS = np.where(np.logical_and(~np.isnan(dedk), abs(e) < 0.1))

for v in dedk[FS].min(), dedk[FS].max(), np.average(dedk[FS]):
    info(int(round(2 * kymax * v / 1e-3)))
