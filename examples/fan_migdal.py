#!/usr/bin/env python3

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import elphmod.models.chain
import matplotlib.pyplot as plt
import numpy as np

nq = 120

kT = 1000 * elphmod.misc.kB / elphmod.misc.Ry
eta = 0.05 / elphmod.misc.Ry

el, ph, elph = elphmod.models.chain.create('data/chain', rydberg=True)

q = np.linspace(0, 2 * np.pi, nq, endpoint=False)
k = q[:nq // 2 + 1]

e = elphmod.dispersion.dispersion(el.H, k)
E, U = elphmod.dispersion.dispersion(el.H, q, vectors=True)
w2, u = elphmod.dispersion.dispersion(ph.D, q, vectors=True)

g2 = np.empty((len(q), ph.size, len(k), el.size, el.size))

for iq in range(len(q)):
    for ik in range(len(k)):
        g2[iq, :, ik, :, :] = abs(np.einsum('xu,am,xab,bn->xmn', u[iq],
            U[(ik + iq) % nq].conj(), elph.g(q1=q[iq], k1=k[ik]), U[ik])) ** 2

w = elphmod.ph.sgnsqrt(w2)
w[0, :3] = eta # avoid division by zero acoustic frequency
g2 /= 2 * w[:, :, np.newaxis, np.newaxis, np.newaxis]

omega = np.linspace(1.5 * e.min(), 1.5 * e.max(), nq)

Sigma = elphmod.diagrams.fan_migdal_self_energy(k.reshape((-1, 1)), E, w, g2,
    omega + 1j * eta, kT)

G = 1 / (omega[np.newaxis, np.newaxis, :] - e[:, :, np.newaxis] - Sigma)

A = -G.imag / np.pi

if elphmod.MPI.comm.rank == 0:
    plt.imshow(A[:, 0, ::-1].T, extent=(k[0], k[-1], omega[0], omega[-1]),
        cmap='plasma')

    plt.plot(k, e, 'w--')

    plt.ylabel('Electron energy (Ry)')
    plt.xlabel('Wave vector')
    plt.xticks([0.0, np.pi], [r'$\Gamma$', 'X'])
    plt.axis('auto')
    plt.show()
