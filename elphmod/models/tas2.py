#!/usr/bin/env python3

# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import numpy as np

import elphmod.bravais
import elphmod.dispersion
import elphmod.el
import elphmod.elph
import elphmod.misc
import elphmod.ph

deg = elphmod.bravais.deg

Npm = (1e-10 * elphmod.misc.a0) ** 2 / (elphmod.misc.eVSI * elphmod.misc.Ry)

a = 3.34 # AA

M = 180.95 * elphmod.misc.uRy
m = 32.06 * elphmod.misc.uRy

e_z2 = 1.85 / elphmod.misc.Ry
e_x2y2 = 2.3 / elphmod.misc.Ry
t_z2 = -0.14 / elphmod.misc.Ry
t_z2_x2y2 = 0.48 / elphmod.misc.Ry
t_z2_xy = -0.38 / elphmod.misc.Ry
t_x2y2 = -0.26 / elphmod.misc.Ry
t_x2y2_xy = 0.31 / elphmod.misc.Ry
t_xy = 0.32 / elphmod.misc.Ry

ax = 43.8 * Npm
ay = 9.0 * Npm
az = 57.3 * Npm
axz = 51.7 * Npm
azx = 39.2 * Npm
by = 3.4 * Npm
bz = 38.1 * Npm
cx = -5.4 * Npm
cy = -4.6 * Npm
cz = -19.0 * Npm
dx = 14.3 * Npm
dy = -1.7 * Npm
dz = -4.8 * Npm
dyz = 2.4 * Npm
ex = 8.0 * Npm
ez = 1.8 * Npm
fx = 2.0 * Npm
fy = -5.6 * Npm
fz = 2.1 * Npm
fyz = 3.0 * Npm

beta = 5.0

at = elphmod.bravais.primitives(ibrav=4, a=a, c=15.0, bohr=True)
r = np.dot([[1.0, 2.0, 0.0], [2.0, 1.0, +0.1], [2.0, 1.0, -0.1]], at)
r[:, :2] /= 3

a /= elphmod.misc.a0

nk = (3, 3, 1) # for electrons
nq = (2, 2, 1) # for phonons
nQ = (3, 3, 1) # for coupling

k = elphmod.bravais.mesh(*nk)
q = elphmod.bravais.mesh(*nq)
Q = elphmod.bravais.mesh(*nQ, flat=True)

e0 = np.diag([e_z2, e_x2y2, e_x2y2])

t0 = np.array([
    [ t_z2,      t_z2_x2y2, t_z2_xy  ],
    [ t_z2_x2y2, t_x2y2,    t_x2y2_xy],
    [-t_z2_xy,  -t_x2y2_xy, t_xy     ],
])

def R2(phi):
    return np.array([
        [1,               0,                0],
        [0, np.cos(2 * phi), -np.sin(2 * phi)],
        [0, np.sin(2 * phi),  np.cos(2 * phi)],
    ])

def hopping(t0, phi):
    return R2(phi) @ t0 @ R2(-phi)

def dR2_dphi(phi):
    return 2 * np.array([
        [0,                0,                0],
        [0, -np.sin(2 * phi), -np.cos(2 * phi)],
        [0,  np.cos(2 * phi), -np.sin(2 * phi)],
    ])

def derivative(t0, phi):
    dt_dr = -beta / a * hopping(t0, phi)

    dt_dphi = dR2_dphi(phi) @ t0 @ R2(-phi) - R2(phi) @ t0 @ dR2_dphi(-phi)

    dt_dx = dt_dr * np.cos(phi) - dt_dphi / a * np.sin(phi)
    dt_dy = dt_dr * np.sin(phi) + dt_dphi / a * np.cos(phi)

    return np.array([dt_dx, dt_dy])

t = np.zeros((6, 3, 3))
dt = np.zeros((6, 9, 3, 3))

for n in range(3):
    t[2 * n] = hopping(t0, n * 120 * deg)
    t[2 * n + 1] = hopping(t0.T, (120 * n + 60) * deg)
    dt[2 * n, :2] = derivative(t0, n * 120 * deg)
    dt[2 * n + 1, :2] = derivative(t0.T, (120 * n + 60) * deg)

def hamiltonian(k1=0.0, k2=0.0, k3=0.0):
    """Calculate electrons according to Eq. (5) of PRB 101, 155107 (2020)."""

    return e0 + (
        + t[0] * np.exp(1j * k1)
        + t[3] * np.exp(-1j * k1)
        + t[1] * np.exp(1j * (k1 + k2))
        + t[4] * np.exp(-1j * (k1 + k2))
        + t[2] * np.exp(1j * k2)
        + t[5] * np.exp(-1j * k2)
    )

def R(phi):
    return np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi),  np.cos(phi), 0],
        [          0,            0, 1],
    ])

def rotate(matrix, phi):
    return R(phi) @ matrix @ R(-phi)

def xreflect(matrix):
    matrix = matrix.copy()

    matrix[0, 1:] *= -1
    matrix[1:, 0] *= -1

    return matrix

def zreflect(matrix):
    matrix = matrix.copy()

    matrix[2, :2] *= -1
    matrix[:2, 2] *= -1

    return matrix

A0 = np.array([
    [ax,  0,  axz],
    [0,   ay, 0  ],
    [azx, 0,  az ],
])

A1 = rotate(A0, -30 * deg)
A2 = rotate(A0, 90 * deg)
A3 = rotate(A0, 210 * deg)

B = np.diag([by, by, bz])

C1 = np.diag([cx, cy, cz])

C3 = rotate(C1, 120 * deg)
C5 = rotate(C1, 240 * deg)

C4 = xreflect(C1)
C2 = xreflect(C3)
C6 = xreflect(C5)

D1 = np.array([
    [ dx, 0,   0  ],
    [ 0,  dy,  dyz],
    [-0,  dyz, dz ],
])

D3 = rotate(D1, 120 * deg)
D5 = rotate(D1, 240 * deg)

D4 = xreflect(D1)
D2 = xreflect(D3)
D6 = xreflect(D5)

E0 = np.diag([ex, ex, ez])

E1 = rotate(E0, 30 * deg)
E2 = rotate(E0, 150 * deg)
E3 = rotate(E0, 270 * deg)

F1 = np.array([
    [fx, 0,   0  ],
    [0,  fy,  fyz],
    [0, -fyz, fz ],
])

F3 = rotate(F1, 120 * deg)
F5 = rotate(F1, 240 * deg)

F4 = xreflect(F1)
F2 = xreflect(F3)
F6 = xreflect(F5)

def dynamical_matrix(q1=0.0, q2=0.0, q3=0.0):
    """Calculate phonons as in Sec. 2.4.3.2 of doi:10.26092/elib/250.

    The DFPT force constants come from Fig. 3 (f-j) of PRX 13, 041009 (2023).
    Load spring "TaS2-SR" on https://janberges.de/spring to visualize them.
    """
    D = np.zeros((9, 9), dtype=complex)

    intra = A1 + A2 + A3
    inter = A1 + A2 * np.exp(1j * q2) + A3 * np.exp(-1j * q1)

    D[0:3, 0:3] += intra
    D[0:3, 3:6] -= inter
    D[3:6, 3:6] += intra.T

    intra = zreflect(intra)
    inter = zreflect(inter)

    D[0:3, 0:3] += intra
    D[0:3, 6:9] -= inter
    D[6:9, 6:9] += intra.T

    D[6:9, 6:9] += B
    D[6:9, 3:6] -= B
    D[3:6, 3:6] += B.T

    D[0:3, 0:3] += (
        + C1 * (1 - np.exp(1j * q1))
        + C2 * (1 - np.exp(1j * (q1 + q2)))
        + C3 * (1 - np.exp(1j * q2))
        + C4 * (1 - np.exp(-1j * q1))
        + C5 * (1 - np.exp(-1j * (q1 + q2)))
        + C6 * (1 - np.exp(-1j * q2))
    )

    both = (
        + D1 * (1 - np.exp(1j * q1))
        + D2 * (1 - np.exp(1j * (q1 + q2)))
        + D3 * (1 - np.exp(1j * q2))
        + D4 * (1 - np.exp(-1j * q1))
        + D5 * (1 - np.exp(-1j * (q1 + q2)))
        + D6 * (1 - np.exp(-1j * q2))
    )

    D[3:6, 3:6] += both
    D[6:9, 6:9] += zreflect(both)

    intra = E1 + E2 + E3
    inter = (
        + E1 * np.exp(1j * (q1 + q2))
        + E2 * np.exp(-1j * (q1 - q2))
        + E3 * np.exp(-1j * (q1 + q2))
    )

    D[0:3, 0:3] += intra
    D[0:3, 3:6] -= inter
    D[3:6, 3:6] += intra.T

    intra = zreflect(intra)
    inter = zreflect(inter)

    D[0:3, 0:3] += intra
    D[0:3, 6:9] -= inter
    D[6:9, 6:9] += intra.T

    intra = F1 + F2 + F3 + F4 + F5 + F6
    inter = (
        + F1 * np.exp(1j * q1)
        + F2 * np.exp(1j * (q1 + q2))
        + F3 * np.exp(1j * q2)
        + F4 * np.exp(-1j * q1)
        + F5 * np.exp(-1j * (q1 + q2))
        + F6 * np.exp(-1j * q2)
    )

    D[6:9, 6:9] += intra
    D[6:9, 3:6] -= inter
    D[3:6, 3:6] += intra.T

    D[3:6, 0:3] = D[0:3, 3:6].conj().T
    D[6:9, 0:3] = D[0:3, 6:9].conj().T
    D[3:6, 6:9] = D[6:9, 3:6].conj().T

    D[0:3, :] /= np.sqrt(M)
    D[:, 0:3] /= np.sqrt(M)

    D[3:9, :] /= np.sqrt(m)
    D[:, 3:9] /= np.sqrt(m)

    return D

sqrtM = np.sqrt(np.repeat([M, m, m], 3)[:, np.newaxis, np.newaxis])

def coupling(q1=0, q2=0, q3=0, k1=0, k2=0, k3=0, **ignore):
    """Calculate coupling according to Eq. (B4) of PRB 101, 155107 (2020)."""

    K1 = k1 + q1
    K2 = k2 + q2

    return (
        + dt[0] * (np.exp(1j * K1) - np.exp(1j * k1))
        + dt[3] * (np.exp(-1j * K1) - np.exp(-1j * k1))
        + dt[1] * (np.exp(1j * (K1 + K2)) - np.exp(1j * (k1 + k2)))
        + dt[4] * (np.exp(-1j * (K1 + K2)) - np.exp(-1j * (k1 + k2)))
        + dt[2] * (np.exp(1j * K2) - np.exp(1j * k2))
        + dt[5] * (np.exp(-1j * K2) - np.exp(-1j * k2))
    ) / sqrtM

def create(prefix=None, rydberg=False, divide_mass=True):
    """Create tight-binding, mass-spring, and coupling data files for TMDCs.

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
    g = elphmod.elph.sample(coupling, Q, nk)

    el = elphmod.el.Model(rydberg=rydberg)
    elphmod.el.k2r(el, H, at, r[:1].repeat(3, axis=0), rydberg=True)
    el.standardize(eps=1e-10)

    ph = elphmod.ph.Model(amass=[M, m, m], at=at, tau=r,
        atom_order=['Ta', 'S', 'S'], divide_mass=divide_mass)
    elphmod.ph.q2r(ph, D_full=D)
    ph.standardize(eps=1e-10)

    elph = elphmod.elph.Model(el=el, ph=ph, divide_mass=divide_mass)
    elphmod.elph.q2r(elph, nQ, nk, g, r=np.repeat(r[:1], el.size, axis=0))
    elph.standardize(eps=1e-10)

    if prefix is not None:
        el.to_hrdat(prefix)
        ph.to_flfrc('%s.ifc' % prefix)
        elph.to_epmatwp(prefix)

    return el, ph, elph
