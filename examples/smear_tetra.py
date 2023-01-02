#!/usr/bin/env python3

# Copyright (C) 2017-2023 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt
import numpy as np

comm = elphmod.MPI.comm

mu = -0.1665

nk = 120
kT = np.linspace(0.1, 0.0, 1000, endpoint=False)

q1 = 0.0
q2 = 0.5

el = elphmod.el.Model('data/NbSe2')

ekk = elphmod.dispersion.dispersion_full(el.H, nk)[:, :, 0] - mu

ekq = np.roll(np.roll(ekk,
    shift=-int(round(q1 * nk)), axis=0),
    shift=-int(round(q2 * nk)), axis=1)

DOS_tetra = elphmod.dos.hexDOS(ekk)(0.0)
DDI_tetra = sum(elphmod.dos.double_delta(ekk, ekq)(0.0).values())

sizes, bounds = elphmod.MPI.distribute(len(kT), bounds=True)

my_DOS_smear = np.empty(sizes[comm.rank])
my_DDI_smear = np.empty(sizes[comm.rank])

progress = elphmod.misc.StatusBar(sizes[comm.rank],
    title='calculate DOS and DDI via smearing')

for my_n, n in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
    delta_kk = elphmod.occupations.fermi_dirac.delta(ekk / kT[n]) / kT[n]
    delta_kq = elphmod.occupations.fermi_dirac.delta(ekq / kT[n]) / kT[n]

    my_DOS_smear[my_n] = np.average(delta_kk)
    my_DDI_smear[my_n] = np.average(delta_kk * delta_kq)

    progress.update()

DOS_smear = np.empty(len(kT))
DDI_smear = np.empty(len(kT))

comm.Gatherv(my_DOS_smear, (DOS_smear, sizes))
comm.Gatherv(my_DDI_smear, (DDI_smear, sizes))

if comm.rank == 0:
    plt.ylabel('DOS (eV$^{-1}$), DDI (eV$^{-2}$)')
    plt.xlabel('Smearing $k T$ (eV)')

    plt.axhline(DOS_tetra, color='r', linestyle='--', label='DOS (tetra.)')
    plt.axhline(DDI_tetra, color='b', linestyle='--', label='DDI (tetra.)')

    plt.plot(kT, DOS_smear, 'r', label='DOS (smear.)')
    plt.plot(kT, DDI_smear, 'b', label='DDI (smear.)')

    plt.legend()
    plt.show()
