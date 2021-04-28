#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import copy
import elphmod
import matplotlib.pyplot as plt
import numpy as np

comm = elphmod.MPI.comm

data = 'NbSe2_cDFPT'

nq = 12

q_path, x = elphmod.bravais.GMKG(301)

colors = ['skyblue', 'dodgerblue', 'orange']

ph = elphmod.ph.Model('data/%s.ifc' % data, apply_asr=True)

q = sorted(elphmod.bravais.irreducibles(nq))
q = np.array(q, dtype=float) / nq * 2 * np.pi

D = elphmod.dispersion.sample(ph.D, q)

ph2 = copy.copy(ph)
elphmod.ph.q2r(ph2, D, q, nq, apply_asr=True)

plt.figure()

for subplot, D in enumerate([ph.D, ph2.D]):
    plt.subplot(121 + subplot)

    w2, e, order = elphmod.dispersion.dispersion(D, q_path,
        vectors=True, order=True, broadcast=False)

    if comm.rank == 0:
        w = elphmod.ph.sgnsqrt(w2) * elphmod.misc.Ry * 1e3

        pol = elphmod.ph.polarization(e, q_path)

        for i in range(w.shape[1]):
            fatbands = elphmod.plot.compline(x, w[:, i], pol[:, i])

            for j, fatband in enumerate(fatbands):
                plt.fill(*fatband, color=colors[j], linewidth=0.0)

if comm.rank == 0:
    plt.show()
