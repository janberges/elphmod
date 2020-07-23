#/usr/bin/env python3

import elphmod
import matplotlib.pyplot as plt
import numpy as np

kT = 0.025

k = np.linspace(0, 2 * np.pi, 36, endpoint=False)
E = np.empty(k.shape * 2)

for i, p in enumerate(k):
    for j, q in enumerate(k):
        E[i, j] = -np.sqrt(3 + 2 * (np.cos(p) + np.cos(q) + np.cos(p + q)))

e = np.linspace(E.min(), E.max(), 300)

DOS_smear = np.empty(len(e))

for n in range(len(e)):
    x = (E - e[n]) / kT

    DOS_smear[n] = np.average(elphmod.occupations.fermi_dirac.delta(x)) / kT

DOS_tetra = elphmod.dos.hexDOS(E)(e)

if elphmod.MPI.comm.rank == 0:
    plt.xlabel('energy (eV)')
    plt.ylabel('density of states (1/eV)')
    plt.fill_between(e, 0.0, DOS_tetra, facecolor='lightgray', label='tetra.')
    plt.plot(e, DOS_smear, color='red', label='smear.')
    plt.legend()
    plt.show()
