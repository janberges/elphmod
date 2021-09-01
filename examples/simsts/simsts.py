#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt
import numpy as np

el = elphmod.el.Model('graphene')

e, U, order = elphmod.dispersion.dispersion_full(el.H, 72,
    vectors=True, order=True)

e -= elphmod.el.read_Fermi_level('scf.out')

weight = (U.conj() * U)[:, :, :2].real.sum(axis=2)

w = np.linspace(e.min(), e.max(), 150)

DOS = 0

for n in range(el.size):
    DOS = DOS + elphmod.dos.hexa2F(e[:, :, n], weight[:, :, n])(w)

if elphmod.MPI.comm.rank == 0:
    plt.xlabel('energy (eV)')
    plt.ylabel('density of states (1/eV)')
    plt.fill(w, DOS, color='black', linewidth=0.0)
    plt.show()
