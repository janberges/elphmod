#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import numpy as np

nbnd = 3
offset = 6

x, k, eps, proj = elphmod.el.read_atomic_projections(
    'work/polonium.save/atomic_proj.xml')

orbitals = elphmod.el.read_projwfc_out('projwfc.out')
weight = elphmod.el.proj_sum(proj, orbitals, 'p')

if elphmod.MPI.comm.rank != 0:
    raise SystemExit

subspace = np.empty((nbnd, len(k)), dtype=int)

for ik in range(len(k)):
    subspace[:, ik] = np.sort(np.argsort(weight[ik, :, 0])[-nbnd:])

nk = int(np.cbrt(len(k)))

subspace = subspace.reshape((nbnd, nk, nk, nk))

with open('subspace.in', 'w') as text:
    text.write('%d %d %d %d %d\n\n' % (nk, nk, nk, nbnd, offset))
    for n in range(subspace.shape[0]):
        for k1 in range(subspace.shape[1]):
            for k2 in range(subspace.shape[2]):
                for k3 in range(subspace.shape[3]):
                    text.write('%2d' % (subspace[n, k1, k2, k3] + 1 - offset))
                text.write('\n')
            text.write('\n')
        text.write('\n')
