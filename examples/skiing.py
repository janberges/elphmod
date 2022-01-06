#!/usr/bin/env python3

# Copyright (C) 2017-2022 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

symmetric = True

ph = elphmod.ph.Model('data/NbSe2_DFPT.ifc', divide_mass=False)
ph = ph.supercell(3, 3)

for S in 6, 8, 15, 17, 24, 26:
    ph.shift_atoms(S, (0, -1, 0))

C = ph.D()

def E2(u):
    u /= np.linalg.norm(u)
    return np.einsum('x,xy,y', u, C, u).real

if symmetric:
    u = np.zeros(ph.size)
    Nb3 = [4, 13, 16]
    u.reshape(ph.r.shape)[Nb3] = np.average(ph.r[Nb3], axis=0) - ph.r[Nb3]
else:
    u = 1 - 2 * np.random.rand(ph.size)

u = scipy.optimize.minimize(E2, u).x
u /= u.max()

if elphmod.MPI.comm.rank == 0:
    Nb = np.array([X == 'Nb' for X in ph.atom_order])

    ax = plt.axes(projection='3d')
    ax.set_box_aspect(np.ptp(ph.r, axis=0))
    ax.set_axis_off()

    ax.scatter(*ph.r[Nb].T, s=100.0, c='k')
    ax.scatter(*ph.r[~Nb].T, s=50.0, c='y')

    ax.quiver(*ph.r.T, *u.reshape(ph.r.shape).T)

    for n in range(ph.nat):
        ax.text(*ph.r[n], str(n))

    plt.show()
