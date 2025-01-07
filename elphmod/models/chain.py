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

a = 1.0 # AA
c = 20.0 # AA

M = elphmod.misc.uRy

t = 0.5 / elphmod.misc.Ry
w0 = 0.05 / elphmod.misc.Ry
g0 = 0.02 / elphmod.misc.Ry ** 1.5

at = elphmod.bravais.primitives(ibrav=8, a=a, b=c, c=c, bohr=True)
r = np.zeros((1, 3))

nk = (3, 1, 1) # for electrons
nq = (2, 1, 1) # for phonons
nQ = (3, 1, 1) # for coupling

k = elphmod.bravais.mesh(*nk)
q = elphmod.bravais.mesh(*nq)
Q = elphmod.bravais.mesh(*nQ, flat=True)

def hamiltonian(k1=0, k2=0, k3=0):
    """Calculate electrons as in Eq. (62) of PRX 13, 041009 (2023)."""

    H = np.empty((1, 1))

    H[0, 0] = -2 * t * np.cos(k1)

    return H

def dynamical_matrix(q1=0, q2=0, q3=0):
    """Calculate phonons as in Eq. (63) of PRX 13, 041009 (2023)."""

    D = np.eye(3)

    D *= w0 ** 2 * (1 - np.cos(q1))

    return D

def coupling(q1=0, q2=0, q3=0, k1=0, k2=0, k3=0, **ignore):
    """Calculate el.-ph. coupling as in Eq. (64) of PRX 13, 041009 (2023)."""

    g = np.zeros((3, 1, 1), dtype=complex)

    g[0, 0, 0] = 1j * g0 * (np.sin(k1 + q1) - np.sin(k1))

    return g

def create(prefix=None, rydberg=False, divide_mass=True):
    """Create tight-binding, mass-spring, and coupling data files for chain.

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
    elphmod.el.k2r(el, H, at, r, rydberg=True)
    el.standardize(eps=1e-10)

    ph = elphmod.ph.Model(amass=[M], at=at, tau=r, atom_order=['X'],
        divide_mass=divide_mass)
    elphmod.ph.q2r(ph, D_full=D)
    ph.standardize(eps=1e-10)

    elph = elphmod.elph.Model(el=el, ph=ph, divide_mass=divide_mass)
    elphmod.elph.q2r(elph, nQ, nk, g, r)
    elph.standardize(eps=1e-10)

    if prefix is not None:
        el.to_hrdat(prefix)
        ph.to_flfrc('%s.ifc' % prefix)
        elph.to_epmatwp(prefix)

    return el, ph, elph
