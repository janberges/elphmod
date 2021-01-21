#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt
import numpy as np

comm = elphmod.MPI.comm

mu = -0.1665

nk = 120
dk = 12
nq = nk

kT = 0.01

cmap = elphmod.plot.colormap(
    (0.0, elphmod.plot.Color(0.0, 1, 255, model='PSV')),
    (1.0, elphmod.plot.Color(5.5, 1, 255, model='PSV')))

el = elphmod.el.Model('data/NbSe2_hr.dat')

ekk = elphmod.dispersion.dispersion_full(el.H, nk)[:, :, 0] - mu
Ekk = ekk[::dk, ::dk]

delta_kk = elphmod.occupations.fermi_dirac.delta(ekk / kT) / kT

q = np.array(sorted(elphmod.bravais.irreducibles(nq)))

wedge = np.empty((len(q), 2))

progress = elphmod.misc.StatusBar(len(q),
    title='calculate double-delta integrals')

for iq, (q1, q2) in enumerate(q):
    ekq = np.roll(np.roll(ekk, shift=-q1, axis=0), shift=-q2, axis=1)
    Ekq = ekq[::dk, ::dk]

    delta_kq = elphmod.occupations.fermi_dirac.delta(ekq / kT) / kT

    intersections = elphmod.dos.double_delta(Ekk, Ekq)(0.0)

    wedge[iq, 0] = np.average(delta_kk * delta_kq)
    wedge[iq, 1] = sum(intersections.values())

    progress.update()

mesh = np.empty((nq, nq, 2))

for iq, (q1, q2) in enumerate(q):
    for q1, q2 in elphmod.bravais.images(q1, q2, nq):
        mesh[q1, q2] = wedge[iq]

if comm.rank == 0:
    figure, axes = plt.subplots(1, 2)

for n in range(2):
    BZ = elphmod.plot.toBZ(mesh[..., n], outside=np.nan, points=300)

    image = elphmod.plot.color(BZ, cmap=cmap, maximum=5.0).astype(int)

    if comm.rank == 0:
        axes[n].imshow(image)
        axes[n].axis('image')
        axes[n].axis('off')

if comm.rank == 0:
    plt.show()
