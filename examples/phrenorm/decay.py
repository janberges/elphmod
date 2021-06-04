#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

# Based on code by Arne Schobert.

import elphmod
import matplotlib.pyplot as plt

pwi = elphmod.bravais.read_pwi('scf.in')

if elphmod.MPI.comm.rank != 0:
    raise SystemExit

R, H = elphmod.el.read_decayH('decay.H')
plt.plot(R, H, 'o', color='blue', markersize=10, label='EPW output')

R, H = elphmod.el.decayH('TaS2_hr.dat', **pwi)
plt.plot(R, H, 'o', color='orange', label='calculated from Wannier90 data')

plt.xlabel = 'Distance (angstrom)'
plt.ylabel = 'Hopping (eV)'
plt.legend()
plt.show()
