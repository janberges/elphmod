#!/usr/bin/env python3

# Copyright (C) 2017-2022 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt

elel = elphmod.elel.Model('data/U.ijkl', nq=2, no=1)

q, x, GMKG = elphmod.bravais.path('GMKG', ibrav=4, N=150)

W = elphmod.dispersion.dispersion(elel.W, q)

if elphmod.MPI.comm.rank == 0:
    plt.plot(x, W)
    plt.xticks(x[GMKG], 'GMKG')
    plt.xlabel('wave vector')
    plt.ylabel('Coulomb interaction (eV)')
    plt.show()
