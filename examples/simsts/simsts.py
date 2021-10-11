#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt
import numpy as np

comm = elphmod.MPI.comm
info = elphmod.MPI.info

nk = 72

info('Set up and diagonalize Wannier Hamiltonian..')

el = elphmod.el.Model('graphene', read_xsf=True, normalize_wf=True)

_, order = elphmod.dispersion.dispersion_full(el.H, nk, order=True)
e, U = elphmod.dispersion.dispersion_full_nosym(el.H, nk, vectors=True)

for k1 in range(nk):
    for k2 in range(nk):
        e[k1, k2] = e[k1, k2, order[k1, k2]]
        for n in range(el.size):
            U[k1, k2, n] = U[k1, k2, n, order[k1, k2]]

e -= elphmod.el.read_Fermi_level('scf.out')

info('Set up Bravais lattice vectors..')

pwi = elphmod.bravais.read_pwi('scf.in')
a = elphmod.bravais.primitives(**pwi)

info('Calculate density of states..')

w, dw = np.linspace(e.min(), e.max(), 300, retstep=True)

DOS = sum(elphmod.dos.hexDOS(e[:, :, n])(w) for n in range(el.size))

if comm.rank == 0:
    plt.plot(w, DOS, label='DOS')

info('Calculate scanning-tunneling spectrum..')

position1 = np.array([0.0, 0.0, 3.0]) + (el.tau[0] + el.tau[1]) / 2
position2 = np.array([0.0, 0.0, 3.0]) + el.tau[0]

for position, label in (position1, 'bond'), (position2, 'atom'):
    label = '$%g\,\mathrm{\AA}$ above %s' % (position[2], label)
    info('Position tip %s..' % label)

    if comm.rank == 0:
        cells = range(-3, 4)
        R = np.array([(n1, n2, 0) for n1 in cells for n2 in cells])

        overlap = np.empty((len(R), el.size))
        norm = np.sqrt(np.pi * elphmod.misc.a0 ** 3)

        for iR in range(len(R)):
            shift = np.dot(R[iR], a)
            d = np.linalg.norm(position - shift - el.r, axis=3)
            s = np.exp(-d / elphmod.misc.a0) / norm

            for n in range(el.size):
                overlap[iR, n] = np.sum(s * el.W[n]) * el.dV

    info('Calculate weight of orbitals..')

    weight = np.empty(e.shape)

    if comm.rank == 0:
        scale = 2 * np.pi / nk

        for k1 in range(nk):
            for k2 in range(nk):
                tmp = 0.0
                for iR in range(len(R)):
                    tmp += np.dot(overlap[iR], U[k1, k2]) * np.exp(1j
                        * (k1 * R[iR, 0] + k2 * R[iR, 1]) * scale)
                weight[k1, k2] = (abs(tmp) / nk) ** 2

    comm.Bcast(weight)

    info('Calculate weighted density of states..')

    STS = sum(elphmod.dos.hexa2F(e[:, :, n], weight[:, :, n])(w)
        for n in range(el.size))

    STS *= el.size / (np.sum(STS) * dw)

    if comm.rank == 0:
        plt.plot(w, STS, label='STS %s' % label)

if comm.rank == 0:
    plt.grid()
    plt.xlabel('energy (eV)')
    plt.ylabel('density of states (1/eV)')
    plt.legend()
    plt.show()
