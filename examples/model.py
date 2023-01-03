#!/usr/bin/env python3

# Copyright (C) 2017-2023 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import numpy as np
import elphmod

u = elphmod.misc.uSI / (2 * elphmod.misc.meSI)
Npm = (1e-10 * elphmod.misc.a0) ** 2 / (elphmod.misc.eVSI * elphmod.misc.Ry)

a = 2.46 # AA

M = 12.011 * u

t = -2.6 / elphmod.misc.Ry

Cx = -365.0 * Npm
Cy = -245.0 * Npm
Cz = -98.2 * Npm

# Force constants from Jishi et al., Chem. Phys. Lett. 209, 77 (1993)

beta = 2.0

hr_dat = 'data/graphene_hr.dat'
flfrc = 'data/graphene.ifc'
epmatwp = 'data/graphene.epmatwp'
wigner = 'data/graphene.wigner'

at = elphmod.bravais.primitives(ibrav=4, a=a, c=15.0, bohr=True)
r = np.dot([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0]], at) / 3

nk = (6, 6, 1)
nq = (2, 2, 1)

k = [[[(k1, k2, k3)
    for k3 in range(nk[2])]
    for k2 in range(nk[1])]
    for k1 in range(nk[0])]

q = [[[(q1, q2, q3)
    for q3 in range(nq[2])]
    for q2 in range(nq[1])]
    for q1 in range(nq[0])]

k = 2 * np.pi * np.array(k, dtype=float) / nk
q = 2 * np.pi * np.array(q, dtype=float) / nq

def hamiltonian(k1=0, k2=0, k3=0):
    H = np.zeros((2, 2), dtype=complex)

    H[0, 1] = t * (np.exp(1j * k1) + 1 + np.exp(-1j * k2))
    H[1, 0] = H[0, 1].conj()

    return H

def rotate(A, phi):
    phi *= np.pi / 180

    c = np.cos(phi)
    s = np.sin(phi)

    R = np.array([[c, -s, 0], [s,  c, 0], [0, 0, 1]])

    return R.dot(A).dot(R.T)

K = np.diag([Cx, Cy, Cz])

K1 = rotate(K, 30)
K2 = rotate(K, 150)
K3 = rotate(K, 270)

L = np.diag([0, -Cy / 6, -Cz / 6])

L1 = rotate(L, 0)
L2 = rotate(L, 120)
L3 = rotate(L, 240)

def dynamical_matrix(q1=0, q2=0, q3=0):
    """Calculate phonons shown in Fig. 2.9 of doi:10.26092/elib/250."""

    C = np.empty((6, 6), dtype=complex)

    C[3:, :3] = K1 * np.exp(1j * q1) + K2 + K3 * np.exp(-1j * q2)
    C[:3, :3] = (L1 * np.exp(1j * q1) + L2 * np.exp(1j * q2)
        + L3 * np.exp(-1j * (q1 + q2)))
    C[:3, :3] += C[:3, :3].conj()

    C[:3, :3] -= K1 + K2 + K3 + 2 * (L1 + L2 + L3)

    C[:3, 3:] = C[3:, :3].conj().T
    C[3:, 3:] = C[:3, :3].conj()

    return C / M

tau0 = r[1] - r[0]
tau1 = r[0]
tau2 = -r[1]
tau = np.linalg.norm(tau0)

def coupling(q1=0, q2=0, q3=0, k1=0, k2=0, k3=0, **ignore):
    """Calculate el.-ph. coupling as in Section S9 of arXiv:2108.01121."""

    d = np.zeros((6, 2, 2), dtype=complex)

    K1 = k1 + q1
    K2 = k2 + q2

    d[:3, 0, 1] = tau0 + tau1 * np.exp(1j * k1) + tau2 * np.exp(-1j * k2)
    d[:3, 1, 0] = tau0 + tau1 * np.exp(-1j * K1) + tau2 * np.exp(1j * K2)
    d[3:] = -d[:3].swapaxes(1, 2).conj()

    return beta * t / (tau ** 2 * np.sqrt(M)) * d

H = elphmod.dispersion.sample(hamiltonian, k)
D = elphmod.dispersion.sample(dynamical_matrix, q)
g = elphmod.elph.sample(coupling, q.reshape((-1, 3)), nk)

elphmod.bravais.Fourier_interpolation(H * elphmod.misc.Ry, hr_file=hr_dat)
el = elphmod.el.Model(hr_dat)

ph = elphmod.ph.Model(phid=np.empty((2, 2) + nq + (3, 3)),
    amass=[M] * 2, at=at, tau=r, atom_order=['C'] * 2)

elphmod.ph.q2r(ph, D_full=D, flfrc=flfrc)

Rk, dk, lk = elphmod.bravais.wigner_seitz_x('q', nk[0], at, r)
Rg, dg, lg = elphmod.bravais.wigner_seitz_x('q', nq[0], at, r)

Rk = np.insert(Rk, obj=2, values=0, axis=1)
Rg = np.insert(Rg, obj=2, values=0, axis=1)
dg = dg.swapaxes(0, 1).reshape((1, el.size, ph.nat, len(Rg)))

elph = elphmod.elph.Model(Rk=Rk, dk=dk, Rg=Rg, dg=dg, el=el, ph=ph,
    divide_mass=False)
elphmod.elph.q2r(elph, nq, nk, g)
elph.standardize(eps=1e-10)

if elphmod.MPI.comm.rank == 0:
    dk = np.ones((ph.nat, ph.nat, len(elph.Rk)), dtype=int)
    dg = np.ones((ph.nat, len(elph.Rg), 1, el.size), dtype=int)

    with open(wigner, 'wb') as data:
        for obj in [el.size, ph.nat, len(elph.Rk), elph.Rk, dk,
                len(elph.Rg), elph.Rg, dg]:
            np.array(obj, dtype=np.int32).tofile(data)

    with open(epmatwp, 'wb') as data:
        np.swapaxes(elph.data, 3, 4).astype(np.complex128).tofile(data)
