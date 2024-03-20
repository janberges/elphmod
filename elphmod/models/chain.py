#!/usr/bin/env python3

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import numpy as np
import elphmod

a = 1.0 # AA
c = 20.0 # AA

M = elphmod.misc.uRy

t = 1.0 / elphmod.misc.Ry
w0 = 0.05 / elphmod.misc.Ry
g0 = 0.02 / elphmod.misc.Ry ** 1.5

at = elphmod.bravais.primitives(ibrav=8, a=a, b=c, c=c, bohr=True)
r = np.zeros((1, 3))

nk = (3, 1, 1) # for electrons
nq = (2, 1, 1) # for phonons
nQ = (3, 1, 1) # for coupling

k = [[[(k1, k2, k3)
    for k3 in range(nk[2])]
    for k2 in range(nk[1])]
    for k1 in range(nk[0])]

q = [[[(q1, q2, q3)
    for q3 in range(nq[2])]
    for q2 in range(nq[1])]
    for q1 in range(nq[0])]

Q = [(Q1, Q2, Q3)
    for Q1 in range(nQ[0])
    for Q2 in range(nQ[1])
    for Q3 in range(nQ[2])]

k = 2 * np.pi * np.array(k, dtype=float) / nk
q = 2 * np.pi * np.array(q, dtype=float) / nq
Q = 2 * np.pi * np.array(Q, dtype=float) / nQ

def hamiltonian(k1=0, k2=0, k3=0):
    """Calculate electrons as in Eq. (50) of arXiv:2212.11806."""

    H = np.empty((1, 1))

    H[0, 0] = -t * np.cos(k1)

    return H

def dynamical_matrix(q1=0, q2=0, q3=0):
    """Calculate phonons as in Eq. (51) of arXiv:2212.11806."""

    D = np.eye(3)

    D *= w0 ** 2 * (1 - np.cos(q1))

    return D

def coupling(q1=0, q2=0, q3=0, k1=0, k2=0, k3=0, **ignore):
    """Calculate el.-ph. coupling as in Eq. (52) of arXiv:2212.11806."""

    g = np.zeros((3, 1, 1), dtype=complex)

    g[0, 0, 0] = 1j * g0 * (np.sin(k1) - np.sin(k1 + q1))

    return g

def create(prefix='chain'):
    """Create tight-binding, mass-spring, and coupling data files for chain.

    Parameters
    ----------
    prefix : str, default 'chain'
        Common prefix or seedname of data files.

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

    el = elphmod.el.Model()
    el.size = H.shape[-1]
    elphmod.el.k2r(el, H * elphmod.misc.Ry, at, r)
    el.standardize(eps=1e-10)
    el.to_hrdat(prefix)

    ph = elphmod.ph.Model(phid=np.empty((1, 1) + nq + (3, 3)),
        amass=[M], at=at, tau=r, atom_order=['X'])

    elphmod.ph.q2r(ph, D_full=D)
    ph.standardize(eps=1e-10)
    ph.to_flfrc('%s.ifc' % prefix)

    Rk = np.array([(-1, 0, 0), (0, 0, 0), (1, 0, 0)])
    Rg = Rk.copy()

    dk = np.ones((ph.nat, ph.nat, len(Rk)), dtype=int)
    dg = np.ones((1, el.size, ph.nat, len(Rg)), dtype=int)

    elph = elphmod.elph.Model(Rk=Rk, dk=dk, Rg=Rg, dg=dg, el=el, ph=ph,
        divide_mass=False)
    elphmod.elph.q2r(elph, nQ, nk, g)
    elph.standardize(eps=1e-10)

    if elphmod.MPI.comm.rank == 0:
        dk = np.ones((ph.nat, ph.nat, len(elph.Rk)), dtype=int)
        dg = np.ones((ph.nat, len(elph.Rg), 1, el.size), dtype=int)

        with open('%s.wigner' % prefix, 'wb') as data:
            for obj in [1, 1,
                    len(elph.Rk), elph.Rk, np.ones(len(elph.Rk), dtype=int),
                    len(elph.Rg), elph.Rg, np.ones(len(elph.Rg), dtype=int)]:
                np.array(obj, dtype=np.int32).tofile(data)

        with open('%s.epmatwp' % prefix, 'wb') as data:
            np.swapaxes(elph.data, 3, 4).astype(np.complex128).tofile(data)

    return el, ph, elph
