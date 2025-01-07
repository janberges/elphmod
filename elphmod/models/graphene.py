#!/usr/bin/env python3

# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import numpy as np

import elphmod.bravais
import elphmod.dispersion
import elphmod.el
import elphmod.elel
import elphmod.elph
import elphmod.misc
import elphmod.ph

Npm = (1e-10 * elphmod.misc.a0) ** 2 / (elphmod.misc.eVSI * elphmod.misc.Ry)

a = 2.46 # AA

M = 12.011 * elphmod.misc.uRy

t = -2.6 / elphmod.misc.Ry

Cx = -365.0 * Npm
Cy = -245.0 * Npm
Cz = -98.2 * Npm

# Force constants from Jishi et al., Chem. Phys. Lett. 209, 77 (1993)

beta = 2.0

at = elphmod.bravais.primitives(ibrav=4, a=a, c=15.0, bohr=True)
r = np.dot([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0]], at) / 3

nk = (2, 2, 1)
nq = (2, 2, 1)

k = elphmod.bravais.mesh(*nk)
q = elphmod.bravais.mesh(*nq)

def hamiltonian(k1=0, k2=0, k3=0):
    """Calculate electrons of, e.g., Phys. Rev. B 90, 085422 (2014)."""

    H = np.zeros((2, 2), dtype=complex)

    H[0, 1] = t * (np.exp(1j * k1) + 1 + np.exp(-1j * k2))
    H[1, 0] = H[0, 1].conj()

    return H

def rotate(A, phi):
    phi *= np.pi / 180

    c = np.cos(phi)
    s = np.sin(phi)

    R = np.array([[c, -s, 0], [s,  c, 0], [0, 0, 1]])

    return R @ A @ R.T

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
    """Calculate el.-ph. coupling as in Sec. S9 of PRB 105, L241109 (2022)."""

    d = np.zeros((6, 2, 2), dtype=complex)

    K1 = k1 + q1
    K2 = k2 + q2

    d[:3, 0, 1] = tau0 + tau1 * np.exp(1j * k1) + tau2 * np.exp(-1j * k2)
    d[:3, 1, 0] = tau0 + tau1 * np.exp(-1j * K1) + tau2 * np.exp(1j * K2)
    d[3:] = -d[:3].swapaxes(1, 2).conj()

    return beta * t / (tau ** 2 * np.sqrt(M)) * d

U00 = 9.3 / elphmod.misc.Ry
U01 = 5.5 / elphmod.misc.Ry
U02 = 4.1 / elphmod.misc.Ry
U03 = 3.6 / elphmod.misc.Ry

def coulomb_interaction(q1=0, q2=0, q3=0):
    """Calculate Coulomb interaction of Phys. Rev. Lett. 106, 236805 (2011)."""

    U = np.empty((2, 2), dtype=complex)

    U[0, 0] = U00 + 2 * U02 * (np.cos(q1) + np.cos(q1 + q2) + np.cos(q2))
    U[0, 1] = (U01 * (np.exp(1j * q1) + 1 + np.exp(-1j * q2))
        + U03 * (2 * np.cos(q1 + q2) + np.exp(1j * (q1 - q2))))

    U[1, 1] = U[0, 0]
    U[1, 0] = U[0, 1].conj()

    return U

def create(prefix=None, rydberg=False, divide_mass=True):
    """Create tight-binding, mass-spring, and coupling data files for graphene.

    Parameters
    ----------
    prefix : str, optional
        Common prefix or seedname of data files. If absent, no data is written.
    rydberg : bool, default False
        Store tight-binding model in Ry rather than eV?
    divide_mass : bool, default True
        Divide force constants and electron-phonon coupling by atomic masses and
        their square root, respectively?

    Returns
    -------
    object
        Tight-binding model.
    object
        Mass-spring model.
    object
        Localized electron-phonon coupling.
    object
        Localized electron-electron interaction.
    """
    H = elphmod.dispersion.sample(hamiltonian, k)
    D = elphmod.dispersion.sample(dynamical_matrix, q)
    g = elphmod.elph.sample(coupling, q.reshape((-1, 3)), nk)
    U = elphmod.dispersion.sample(coulomb_interaction, q)

    el = elphmod.el.Model(rydberg=rydberg)
    elphmod.el.k2r(el, H, at, r, rydberg=True)
    el.standardize(eps=1e-10)

    ph = elphmod.ph.Model(amass=[M] * 2, at=at, tau=r, atom_order=['C'] * 2,
        divide_mass=divide_mass)
    elphmod.ph.q2r(ph, D_full=D)
    ph.standardize(eps=1e-10)

    elph = elphmod.elph.Model(el=el, ph=ph, divide_mass=divide_mass)
    elphmod.elph.q2r(elph, nq, nk, g, r)
    elph.standardize(eps=1e-10)

    elel = elphmod.elel.Model()
    elphmod.elel.q2r(elel, U * elphmod.misc.Ry, at, r)
    elel.standardize(eps=1e-10)

    if prefix is not None:
        el.to_hrdat(prefix)
        ph.to_flfrc('%s.ifc' % prefix)
        elph.to_epmatwp(prefix)
        elel.to_Wmat('%s.Wmat' % prefix)

    return el, ph, elph, elel
