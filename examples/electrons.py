#/usr/bin/env python

import elphmod

import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
comm = MPI.COMM_WORLD

eF = -0.1665

if comm.rank == 0:
    print("Set up Wannier Hamiltonian..")

H = elphmod.electrons.hamiltonian(comm, 'data/NbSe2_hr.dat')

if comm.rank == 0:
    print("Diagonalize Hamiltonian along G-M-K-G..")

q, x, GMKG = elphmod.bravais.GMKG(corner_indices=True)

eps, psi, order = elphmod.dispersion.dispersion(comm, H, q,
    vectors=True, order=True)

eps -= eF

if comm.rank == 0:
    print("Diagonalize Hamiltonian on uniform mesh..")

nk = 120

eps_full = elphmod.dispersion.dispersion_full(comm, H, nk) - eF

if comm.rank == 0:
    print("Calculate DOS of metallic band..")

ne = 300

e = np.linspace(eps_full[:, :, 0].min(), eps_full[:, :, 0].max(), ne)

DOS = elphmod.dos.hexDOS(eps_full[:, :, 0], comm)(e)

if comm.rank == 0:
    print("Plot dispersion and DOS..")

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    ax1.set_ylabel('energy (eV)')
    ax1.set_xlabel('wave vector')
    ax2.set_xlabel('density of states (1/eV)')

    ax1.set_xticks(x[GMKG])
    ax1.set_xticklabels('GMKG')

    for i in range(H.size):
        X, Y = elphmod.plot.compline(x, eps[:, i],
            0.05 * (psi[:, :, i] * psi[:, :, i].conj()).real)

        for j in range(H.size):
            ax1.fill(X, Y[j], color='RCB'[j])

    ax2.fill(DOS, e, color='C')

    plt.show()
