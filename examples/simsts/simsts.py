#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt
import numpy as np

comm = elphmod.MPI.comm
info = elphmod.MPI.info

info('Set up and diagonalize Wannier Hamiltonian..')

el = elphmod.el.Model('graphene')

e, U, order = elphmod.dispersion.dispersion_full(el.H, 72,
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

V = np.dot(np.cross(a[0], a[1]), a[2])
dV = V / data.size

info('Sample real space..')

if comm.rank == 0:
    x, y, z = [np.linspace(0.0, 1.0, points) for points in data.shape]
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    r = np.zeros(data.shape + (3,))
    r += np.einsum('xyz,i->xyzi', x, a[0])
    r += np.einsum('xyz,i->xyzi', y, a[1])
    r += np.einsum('xyz,i->xyzi', z, a[2])
    r += r0

info('Probe density of states..')

position1 = np.array([0.0, 0.0, 0.5]) + (tau[0] + tau[1]) / 2
position2 = np.array([0.0, 0.0, 0.5]) + tau[0]

for position, label in (position1, 'bond'), (position2, 'atom'):
    label = '$%g\,\mathrm{\AA}$ above %s' % (position[2], label)
    info('Position tip %s..' % label)

    if comm.rank == 0:
        tip = np.zeros(data.shape)

        sigma = 0.25
        sigma *= np.sqrt(2)

        shifts = [-1, 0, 1]

        for d1 in shifts:
            for d2 in shifts:
                for d3 in shifts:
                    shift = d1 * a1 + d2 * a2 + d3 * a3

                    d = np.linalg.norm(position + shift - r, axis=3)

                    tip += elphmod.occupations.gauss.delta(d / sigma) / sigma

    info('Calculate weight of orbitals..')

    if comm.rank == 0:
        weight = (U.conj() * U).real

        for n in range(el.size):
            factor = np.sum(tip * W[n]) ** 2 * dV
            weight[:, :, n, :] *= factor

            print('%2d %9.3f' % (n + 1, factor))

        weight = weight.sum(axis=2)
    else:
        weight = np.empty(e.shape)

    comm.Bcast(weight)

    info('Calculate weighted density of states..')

    w = np.linspace(e.min(), e.max(), 150)

    DOS = 0

    for n in range(el.size):
        DOS = DOS + elphmod.dos.hexa2F(e[:, :, n], weight[:, :, n])(w)

    if comm.rank == 0:
        plt.plot(w, DOS, label=label)

if comm.rank == 0:
    plt.xlabel('energy (eV)')
    plt.ylabel('density of states (1/eV)')
    plt.legend()
    plt.show()
