#!/usr/bin/env python3

# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import numpy as np

info = elphmod.MPI.info

pwi = elphmod.bravais.read_pwi('pw.in')

nk = np.array(pwi['k_points'][:3])

a = elphmod.bravais.primitives(**pwi)
a /= np.linalg.norm(a[0])
b = elphmod.bravais.reciprocals(*a)

el = elphmod.el.Model('TaS2')
ph = elphmod.ph.Model('dyn')
elph = elphmod.elph.Model('work/TaS2.epmatwp', 'wigner.fmt', el, ph,
    divide_mass=False)

with open('ph.out') as lines:
    for line in lines:
        if 'Calculation of q' in line:
            q = np.array([float(c) for c in line.split()[-3:]])[np.newaxis]

            info('\nq = (%g, %g, %g)\n' % tuple(q[0]))

            q, = 2 * np.pi * elphmod.bravais.cartesian_to_crystal(q, *b)

            info('%2s %2s %2s %2s %2s %2s %9s %9s'
                % ('i', 'm', 'n', 'k1', 'k2', 'k3', 'd (PH)', 'd (EPW)'))

        elif 'Printing the electron-phonon matrix elements' in line:
            next(lines)
            next(lines)

            for line in lines:
                cols = line.split()

                if not cols:
                    break

                i, m, n = tuple(int(c) - 1 for c in cols[:3])
                k1, k2, k3 = tuple(map(int, cols[3:6]))

                k = 2 * np.pi / nk * np.array([k1, k2, k3])

                if m == n == 8:
                    g_ph = float(cols[6]) + 1j * float(cols[7])
                    g_epw = elph.g(*q, *k, elbnd=True, phbnd=False)[i, 0, 0]

                    info('%2d %2d %2d %2d %2d %2d %9.6f %9.6f'
                        % (i, m, n, k1, k2, k3, abs(g_ph), abs(g_epw)))
