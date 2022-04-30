#!/usr/bin/env python3

# Copyright (C) 2017-2022 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

# Based on code by Arne Schobert.

import elphmod
import matplotlib.pyplot as plt

pwi = elphmod.bravais.read_pwi('scf.in')

R1, H1 = elphmod.el.read_decayH('decay.H')
R2, H2 = elphmod.el.decayH('TaS2', **pwi)

if elphmod.MPI.comm.rank == 0:
    plt.plot(R1, H1, 'o', color='blue', markersize=10, label='EPW output')
    plt.plot(R2, H2, 'o', color='orange', label='calculated from Wannier90 data')

    plt.ylabel('Hopping (eV)')
    plt.xlabel(r'Distance ($\mathrm{\AA}$)')
    plt.legend()
    plt.show()
