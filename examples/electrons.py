#/usr/bin/env python

import elphmod

import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
comm = MPI.COMM_WORLD

if comm.rank == 0:
    print("Set up Wannier Hamiltonian..")

H = elphmod.electrons.hamiltonian(comm, 'data/NbSe2_hr.dat')

if comm.rank == 0:
    print("Diagonalize Hamiltonian along G-M-K-G..")

q, x, GMKG = elphmod.bravais.GMKG(corner_indices=True)

eps, psi, order = elphmod.dispersion.dispersion(comm, H, q,
    vectors=True, order=True)

if comm.rank == 0:
    print("Plot dispersion along G-M-K-G..")

    for i in range(psi.shape[2]):
        X, Y = elphmod.plot.compline(x, eps[:, i],
            0.05 * (psi[:, :, i] * psi[:, :, i].conj()).real)

        for j in range(psi.shape[1]):
            plt.fill(X, Y[j], color='RCB'[j])

    plt.xticks(x[GMKG], list('GMKG'))
    plt.show()
