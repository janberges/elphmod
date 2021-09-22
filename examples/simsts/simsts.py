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

el = elphmod.el.Model('graphene')

e, U, order = elphmod.dispersion.dispersion_full(el.H, nk,
    vectors=True, order=True)

e -= elphmod.el.read_Fermi_level('scf.out')

info('Set up Bravais lattice vectors..')

pwi = elphmod.bravais.read_pwi('scf.in')

a1, a2, a3 = elphmod.bravais.primitives(**pwi)

info('Read Wannier functions..')

W = []

for n in range(el.size):
    r0, a, X, tau, data = elphmod.misc.read_xsf('graphene_%05d.xsf' % (n + 1))
    W.append(data)

W = np.array(W)

info('Normalize Wannier functions..')

V = abs(np.dot(np.cross(a[0], a[1]), a[2]))
dV = V / data.size

for n in range(el.size):
    W[n] /= np.sqrt(np.sum(W[n] ** 2) * dV)

info('Check if Wannier functions are orthonormal..')

if comm.rank == 0:
    for m in range(el.size):
        for n in range(el.size):
            print('%2d %2d %7.4f' % (m, n, np.sum(W[m] * W[n]) * dV))

info('Sample real space..')

if comm.rank == 0:
    x, y, z = [np.linspace(0.0, 1.0, points) for points in data.shape]
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    r = np.zeros(data.shape + (3,))
    r += np.einsum('xyz,i->xyzi', x, a[0])
    r += np.einsum('xyz,i->xyzi', y, a[1])
    r += np.einsum('xyz,i->xyzi', z, a[2])
    r += r0

info('Calculate density of states..')

w, dw = np.linspace(e.min(), e.max(), 300, retstep=True)

DOS = sum(elphmod.dos.hexDOS(e[:, :, n])(w) for n in range(el.size))

if comm.rank == 0:
    plt.plot(w, DOS, label='DOS')

info('Calculate scanning-tunneling spectrum..')

position1 = np.array([0.0, 0.0, 3.0]) + (tau[0] + tau[1]) / 2
position2 = np.array([0.0, 0.0, 3.0]) + tau[0]

for position, label in (position1, 'bond'), (position2, 'atom'):
    label = '$%g\,\mathrm{\AA}$ above %s' % (position[2], label)
    info('Position tip %s..' % label)

    if comm.rank == 0:
        tip = np.zeros(data.shape)

        NN = 3
        R = np.array([(n1, n2, 0)
            for n1 in range(-NN, NN + 1)
            for n2 in range(-NN, NN + 1)])

        overlap = np.empty((len(R), el.size))

        norm = np.sqrt(np.pi * elphmod.misc.a0 ** 3)

        for iR, (n1, n2, n3) in enumerate(R):
            shift = n1 * a1 + n2 * a2 + n3 * a3

            d = np.linalg.norm(position - shift - r, axis=3)

            s = np.exp(-d / elphmod.misc.a0) / norm

            for a in range(el.size):
                overlap[iR, a] = np.sum(s * W[a]) * dV

    info('Calculate weight of orbitals..')

    if comm.rank == 0:
        weight = np.zeros(e.shape, dtype=complex)

        scale = 2 * np.pi / nk

        for k1, k2 in sorted(elphmod.bravais.irreducibles(nk)):
            tmp = 0.0
            for iR, (n1, n2, n3) in enumerate(R):
                tmp += np.dot(overlap[iR], U[k1, k2]) * np.exp(-1j
                    * (k1 * n1 + k2 * n2) * scale)

            for K1, K2 in elphmod.bravais.images(k1, k2, nk):
                weight[K1, K2] = tmp

        weight /= nk

        weight = abs(weight) ** 2
    else:
        weight = np.empty(e.shape)

    comm.Bcast(weight)

    info('Calculate weighted density of states..')

    STS = sum(elphmod.dos.hexa2F(e[:, :, n], weight[:, :, n])(w)
        for n in range(el.size))

    STS *= el.size / (np.sum(STS) * dw)

    if comm.rank == 0:
        plt.plot(w, STS, label=label)

if comm.rank == 0:
    plt.grid()
    plt.xlabel('energy (eV)')
    plt.ylabel('density of states (1/eV)')
    plt.legend()
    plt.show()
