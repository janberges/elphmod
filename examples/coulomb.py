#!/usr/bin/env python3

# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt

elel = elphmod.elel.Model('data/U.ijkl', nq=2, no=1)

path = 'GMKG'
q, x, corners = elphmod.bravais.path(path, ibrav=4, N=150)

W = elphmod.dispersion.dispersion(elel.W, q)

if elphmod.MPI.comm.rank == 0:
    plt.plot(x, W)
    plt.ylabel('Coulomb interaction (eV)')
    plt.xlabel('Wave vector')
    plt.xticks(x[corners], path)
    plt.show()
