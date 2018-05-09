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

e, order = elphmod.dispersion.dispersion(comm, H, q, order=True)

if comm.rank == 0:
    print("Plot dispersion along G-M-K-G..")

    plt.xticks(x[GMKG], list('GMKG'))
    plt.plot(x, e)
    plt.show()
