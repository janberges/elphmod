#!/usr/bin/env python3

# Copyright (C) 2017-2022 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import numpy as np
import matplotlib.pyplot as plt

comm = elphmod.MPI.comm
info = elphmod.MPI.info

info('Define parameters')

nk = 60
width = 300
height = 300

kT = 0.01
f = elphmod.occupations.fermi_dirac

eta1 = 0.05 # broadening of self-energy
eta2 = 0.01 # broadening of all modes

cmap = elphmod.plot.colormap(
    (0.00, elphmod.plot.Color(255, 255, 255)),
    (0.05, elphmod.plot.Color(0, 0, 255)),
    (0.10, elphmod.plot.Color(255, 0, 0)),
    (1.00, elphmod.plot.Color(0, 0, 0)),
    )

info('Set up model Hamiltonian from Section S9 of arXiv:2108.01121..')

nel = 2
nph = 4

def hamiltonian(k1=0, k2=0, k3=0):
    H = np.zeros((nel, nel), dtype=complex)

    H[0, 1] = 1 + np.exp(1j * k1) + np.exp(-1j * k2)
    H[1, 0] = H[0, 1].conj()

    return H

def dynamical_matrix(q1=0, q2=0, q3=0):
    D = 3 * np.eye(nph, dtype=complex)

    D[:2, 2:] = -np.eye(2) * (1 + np.exp(1j * q1) + np.exp(-1j * q2))
    D[2:, :2] = D[:2, 2:].transpose().conj()

    return D

tau0 = np.array([-np.sqrt(3), 1.0]) / 2
tau1 = np.array([+np.sqrt(3), 1.0]) / 2
tau2 = np.array([0.0, -1.0])

def deformation_potential(q1=0, q2=0, q3=0, k1=0, k2=0, k3=0, **ignore):
    d = np.zeros((nph, nel, nel), dtype=complex)

    K1 = k1 + q1
    K2 = k2 + q2

    d[:2, 0, 1] = tau0 + tau1 * np.exp(1j * k1) + tau2 * np.exp(-1j * k2)
    d[:2, 1, 0] = tau0 + tau1 * np.exp(-1j * K1) + tau2 * np.exp(1j * K2)
    d[2:] = -d[:2].swapaxes(1, 2).conj()

    return d

info('Sample and diagonalize model Hamiltonian..')

q, x, GMKG = elphmod.bravais.GMKG(nk, corner_indices=True, mesh=True,
    straight=False, lift_degen=False)

e, U = elphmod.dispersion.dispersion_full_nosym(hamiltonian, nk, vectors=True)

D0 = elphmod.dispersion.sample(dynamical_matrix, q)
w0 = elphmod.ph.sgnsqrt(np.linalg.eigvalsh(D0))

g = elphmod.elph.sample(deformation_potential, q, nk, U, broadcast=False)

g2 = elphmod.MPI.SharedArray((len(q), nph, nph, nk, nk, nel, nel),
    dtype=complex)

if comm.rank == 0:
    g2[...] = np.einsum('qiklmn,qjklmn->qijklmn', g.conj(), g)

g2.Bcast()

info('Calculate retarded displacement-displacement correlation function..')

w = np.linspace(w0.min(), w0.max(), len(q))

Pi0 = elphmod.diagrams.phonon_self_energy(q, e, g2, kT,
    occupations=f).reshape((len(q), nph, nph))

Piw = elphmod.diagrams.phonon_self_energy(q, e, g2, kT,
    occupations=f, omega=w + 1j * eta1).reshape((len(q), nph, nph, len(w)))

Dw = D0[..., None] - Pi0[..., None] + Piw

A = elphmod.ph.spectral_function(Dw, w, eta2).T

info('Plot results..')

A = elphmod.plot.adjust_pixels(A, GMKG, x[GMKG], width, height)

A = elphmod.plot.color(A[::-1], minimum=0.0, cmap=cmap).astype(int)

if comm.rank == 0:
    plt.imshow(A, extent=(x.min(), x.max(), w0.min(), w0.max()))
    plt.plot(x, w0, 'k')
    plt.xlabel('wave vector')
    plt.ylabel('phonon frequency (a.u.)')
    plt.xticks(x[GMKG], 'GMKG')
    plt.show()
