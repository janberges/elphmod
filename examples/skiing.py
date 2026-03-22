#!/usr/bin/env python3

# Copyright (C) 2017-2026 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
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

plot = elphmod.plot.AtomsPlot(ph.r, ph.atom_order)
plot.set_displacements(u)
plot.plot('skiing.png', scale=1.0, label=True)
