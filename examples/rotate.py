#/usr/bin/env python

import elphmod
import numpy as np

comm = elphmod.MPI.comm
info = elphmod.MPI.info

np.set_printoptions(precision=5, suppress=True, linewidth=1000)

ph = elphmod.ph.Model('data/NbSe2-cDFPT-SR.ifc', apply_asr=True)

def rotation(phi, n=1):
    block = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi),  np.cos(phi), 0],
        [0,            0,           1],
        ])

    return np.kron(np.eye(n), block)

def reflection(n=1):
    return np.diag([-1, 1, 1] * n)

def apply(A, U):
    return np.einsum('ij,jk,kl->il', U, A, U.T.conj())

a1, a2 = elphmod.bravais.translations()
b1, b2 = elphmod.bravais.reciprocals(a1, a2)

q0 = 0.4123 * b1 - 0.1542 * b2
r0 = ph.r[:, :2].T / ph.a[0, 0]

D0 = ph.D(np.dot(q0, a1), np.dot(q0, a2))

for phi in 0, 2 * np.pi / 3, 4 * np.pi / 3:
    U = rotation(phi)[:2, :2]

    q = np.dot(U, q0)
    r = np.dot(U, r0)

    for reflect in False, True:
        if reflect:
            U = reflection()[:2, :2]

            q = np.dot(U, q)
            r = np.dot(U, r)

        D = ph.D(np.dot(q, a1), np.dot(q, a2))

        D_sym = apply(D0, rotation(phi, 3))

        if reflect:
            D_sym = apply(D_sym, reflection(3))

        phase = np.exp(1j * np.array(np.dot(q, r - r0)))

        for n in range(len(phase)):
            D_sym[3 * n:3 * n + 3, :] *= phase[n].conj()
            D_sym[:, 3 * n:3 * n + 3] *= phase[n]

        info(np.sum(np.absolute(D - D_sym) ** 2) / np.sum(np.absolute(D) ** 2))
