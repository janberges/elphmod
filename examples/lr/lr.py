#!/usr/bin/env python3

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt
import numpy as np
import sys

comm = elphmod.MPI.comm

path = 'KGM'
q, x, corners = elphmod.bravais.path(path, ibrav=4, N=50, moveG=0.1)

if len(sys.argv) > 1 and sys.argv[1] == '--prepare-q':
    if comm.rank == 0:
        q /= 2 * np.pi
        weight = 1 / len(q)

        with open('q.dat', 'w') as filqf:
            filqf.write('%d crystal\n' % len(q))

            for q1, q2, q3 in q:
                filqf.write('%12.10f %12.10f %12.10f %12.10f\n'
                    % (q1, q2, q3, weight))

    raise SystemExit

el = elphmod.el.Model('MoS2')
ph = elphmod.ph.Model('dyn', apply_asr_simple=True, apply_zasr=True)
elph = elphmod.elph.Model('work/MoS2.epmatwp', 'wigner.fmt', el, ph)

g = np.absolute([elph.g(q1, q2, q3, elbnd=True, phbnd=True)
    for q1, q2, q3 in q])

g = np.sort(g, axis=1)

w0, g0 = elphmod.elph.read_prtgkk('epw.out',
    nq=len(q), nmodes=ph.size, nk=1, nbnd=el.size)

if comm.rank == 0:
    plt.plot(x, elphmod.ph.sgnsqrt(2 * w0) * g0[:, :, 0, 0, 0], 'ok')
    plt.plot(x, g[:, :, 0, 0] * (1e3 * elphmod.misc.Ry) ** 1.5, '-k')

    plt.ylabel('Electron-phonon coupling (meV)')
    plt.xlabel('Wave vector')
    plt.xticks(x[corners], path)
    plt.show()
