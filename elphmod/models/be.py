#!/usr/bin/env python3

# Copyright (C) 2017-2026 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import numpy as np

import elphmod.bravais
import elphmod.dispersion
import elphmod.el
import elphmod.elph
import elphmod.misc
import elphmod.ph

deg = elphmod.bravais.deg

a = 2.1 # AA

M = 9.01 * elphmod.misc.uRy

es = 3.0 / elphmod.misc.Ry
ex = 8.1 / elphmod.misc.Ry
ez = 2.7 / elphmod.misc.Ry
ts = -1.6 / elphmod.misc.Ry
tx = 3.1 / elphmod.misc.Ry
ty = 0.4 / elphmod.misc.Ry
tz = -0.7 / elphmod.misc.Ry
tsx = 2.2 / elphmod.misc.Ry

eVpa2 = (elphmod.misc.a0 / a) ** 2 / elphmod.misc.Ry

kx = 15.9 * eVpa2
ky = 3.5 * eVpa2
kz = 1.1 * eVpa2

beta = 0.5

at = elphmod.bravais.primitives(ibrav=4, a=a, c=15.0, bohr=True)
r = np.zeros((1, 3))

a /= elphmod.misc.a0

nk = (3, 3, 1)
nq = (3, 3, 1)

k = elphmod.bravais.mesh(*nk)
q = elphmod.bravais.mesh(*nq)

e0 = np.diag([es, ex, ex, ez])
t0 = np.diag([ts, tx, ty, tz])

t0[0, 1] = tsx
t0[1, 0] = -t0[0, 1]

def R(phi):
    return np.array([
        [1,           0,            0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi),  np.cos(phi), 0],
        [0,           0,            0, 1],
    ])

def hopping(t0, phi):
    return R(phi) @ t0 @ R(-phi)

def dR_dphi(phi):
    return np.array([
        [0,            0,            0, 0],
        [0, -np.sin(phi), -np.cos(phi), 0],
        [0,  np.cos(phi), -np.sin(phi), 0],
        [0,            0,            0, 0],
    ])

def derivative(t0, phi):
    dt_dr = -beta / a * hopping(t0, phi)

    dt_dphi = dR_dphi(phi) @ t0 @ R(-phi) - R(phi) @ t0 @ dR_dphi(-phi)

    dt_dx = dt_dr * np.cos(phi) - dt_dphi / a * np.sin(phi)
    dt_dy = dt_dr * np.sin(phi) + dt_dphi / a * np.cos(phi)

    return np.array([dt_dx, dt_dy])

t = np.empty((6, 4, 4))
dt = np.zeros((6, 3, 4, 4))

for n in range(6):
    t[n] = hopping(t0, n * 60 * deg)
    dt[n, :2] = derivative(t0, n * 60 * deg)

def hamiltonian(k1=0.0, k2=0.0, k3=0.0):
    """Calculate electrons as in Appendix D.2.2 of doi:10.26092/elib/250."""

    return e0 + (
        + t[0] * np.exp(1j * k1)
        + t[3] * np.exp(-1j * k1)
        + t[1] * np.exp(1j * (k1 + k2))
        + t[4] * np.exp(-1j * (k1 + k2))
        + t[2] * np.exp(1j * k2)
        + t[5] * np.exp(-1j * k2)
    )

def R_cart(phi):
    return np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi),  np.cos(phi), 0],
        [          0,            0, 1],
    ])

def rotate(matrix, phi):
    return R_cart(phi) @ matrix @ R_cart(-phi)

K0 = np.diag([kx, ky, kz])

K = np.empty((6, 3, 3))

for n in range(6):
    K[n] = rotate(K0, n * 60 * deg)

def dynamical_matrix(q1=0.0, q2=0.0, q3=0.0):
    """Calculate phonons as in Appendix D.2.2 of doi:10.26092/elib/250."""

    return (
        + K[0] * (1 - np.exp(1j * q1))
        + K[3] * (1 - np.exp(-1j * q1))
        + K[1] * (1 - np.exp(1j * (q1 + q2)))
        + K[4] * (1 - np.exp(-1j * (q1 + q2)))
        + K[2] * (1 - np.exp(1j * q2))
        + K[5] * (1 - np.exp(-1j * q2))
    ) / M

dt_dr = np.zeros((6, 3, 4, 4))

def coupling(q1=0, q2=0, q3=0, k1=0, k2=0, k3=0, **ignore):
    """Calculate coupling as in Appendix D.2.2 of doi:10.26092/elib/250."""

    K1 = k1 + q1
    K2 = k2 + q2

    return (
        + dt[0] * (np.exp(1j * K1) - np.exp(1j * k1))
        + dt[3] * (np.exp(-1j * K1) - np.exp(-1j * k1))
        + dt[1] * (np.exp(1j * (K1 + K2)) - np.exp(1j * (k1 + k2)))
        + dt[4] * (np.exp(-1j * (K1 + K2)) - np.exp(-1j * (k1 + k2)))
        + dt[2] * (np.exp(1j * K2) - np.exp(1j * k2))
        + dt[5] * (np.exp(-1j * K2) - np.exp(-1j * k2))
    ) / np.sqrt(M)

def create(prefix=None, rydberg=False, divide_mass=True):
    """Create tight-binding, mass-spring, and coupling data files for beryllium.

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
    """
    H = elphmod.dispersion.sample(hamiltonian, k)
    D = elphmod.dispersion.sample(dynamical_matrix, q)
    g = elphmod.elph.sample(coupling, q.reshape((-1, 3)), nk)

    el = elphmod.el.Model(rydberg=rydberg)
    elphmod.el.k2r(el, H, at, r.repeat(4, axis=0), rydberg=True)
    el.standardize(eps=1e-10)

    ph = elphmod.ph.Model(amass=[M], at=at, tau=r, atom_order=['Be'],
        divide_mass=divide_mass)
    elphmod.ph.q2r(ph, D_full=D)
    ph.standardize(eps=1e-10)

    elph = elphmod.elph.Model(el=el, ph=ph, divide_mass=divide_mass)
    elphmod.elph.q2r(elph, nq, nk, g, r=r.repeat(el.size, axis=0))
    elph.standardize(eps=1e-10)

    if prefix is not None:
        el.to_hrdat(prefix)
        ph.to_flfrc('%s.ifc' % prefix)
        elph.to_epmatwp(prefix)

    return el, ph, elph
