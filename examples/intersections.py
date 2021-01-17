#!/usr/bin/env python3

# Copyright (C) 2020 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt
import numpy as np

mu = -0.1665

nk = 36

q1 = 0.0
q2 = 0.3

el = elphmod.el.Model('data/NbSe2_hr.dat')

ekk = elphmod.dispersion.dispersion_full(el.H, nk)[:, :, 0] - mu

ekq = np.roll(np.roll(ekk,
    shift=-int(round(q1 * nk)), axis=0),
    shift=-int(round(q2 * nk)), axis=1)

intersections = elphmod.dos.double_delta(ekk, ekq)(0.0)

plot = dict(return_k=True, outside=np.nan, points=100)

kxmax, kymax, kx, ky, FS_kk = elphmod.plot.toBZ(ekk, **plot)
kxmax, kymax, kx, ky, FS_kq = elphmod.plot.toBZ(ekq, **plot)

if elphmod.MPI.comm.rank == 0:
    a1, a2 = elphmod.bravais.translations()
    b1, b2 = elphmod.bravais.reciprocals(a1, a2)

    x, y = zip(*[(k1 * b1 + k2 * b2) / nk
        for k1, k2 in intersections.keys()
        for k1, k2 in elphmod.bravais.to_Voronoi(k1, k2, nk=nk)])

    plt.contour(kx, ky, FS_kk, levels=[0.0], colors='k')
    plt.contour(kx, ky, FS_kq, levels=[0.0], colors='k')
    plt.plot(*zip(*elphmod.bravais.BZ()), color='b')

    plt.plot(x, y, 'ob')
    plt.axis('image')
    plt.axis('off')
    plt.show()
