#!/usr/bin/env python3

# Copyright (C) 2017-2022 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import numpy as np

from . import bravais, dos, misc

def Tc(lamda, wlog, mustar=0.1):
    """Calculate critical temperature using McMillan's formula.

    Parameters
    ----------
    lamda : float
        Effective electron-phonon coupling strength.
    wlog : float
        Effective phonon energy in eV.
    mustar : float
        Coulomb pseudopotential.

    Returns
    -------
    float
        Critical temperature in kelvin.
    """
    return wlog / (1.20 * misc.kB) * np.exp(-1.04 * (1 + lamda)
        / max(1e-3, (lamda - mustar * (1 + 0.62 * lamda))))

def McMillan(nq, e, w2, g2, eps=1e-10, mustar=0.0):
    r"""Calculate parameters and result of McMillan's formula.

    Parameters
    ----------
    nq : int
        Number of q points per dimension.
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    w2 : ndarray
        Squared phonon frequencies for irreducible q points.
    g2 : ndarray
        Squared electron-phonon coupling (energy to the power of three).
    eps : float
        Phonon frequencies squared below `eps` are set to `eps`; corresponding
        coupling are set to zero.
    mustar : float
        Coulomb pseudopotential.

    Returns
    -------
    float
        Effective electron-phonon coupling strength :math:`\lambda`.
    float
        Effective phonon frequency :math:`\langle \omega \rangle`.
    float
        Estimated critical temperature of superconductivity.
    """
    nk, nk, nel = e.shape
    nQ, nph = w2.shape
    nQ, nph, nk, nk, nel, nel = g2.shape

    q = np.array(sorted(bravais.irreducibles(nq)))

    weights = np.array([len(bravais.images(q1, q2, nq)) for q1, q2 in q])

    q *= nk // nq

    g2dd = np.zeros((nQ, nph))
    dd = np.zeros(nQ)

    for iq, (q1, q2) in enumerate(q):
        E = np.roll(np.roll(e, shift=-q1, axis=0), shift=-q2, axis=1)

        g2_fun = bravais.linear_interpolation(g2[iq], axes=(1, 2))

        for n in range(nel):
            for m in range(nel):
                intersections = dos.double_delta(e[:, :, n], E[:, :, m])(0)

                for (k1, k2), weight in intersections.items():
                    g2dd[iq] += weight * g2_fun(k1, k2)[:, n, m]
                    dd[iq] += weight

    N0 = 0

    for n in range(nel):
        N0 += dos.hexDOS(e[:, :, n])(0)

    w2 = w2.copy()

    dangerous = np.where(w2 < eps)

    w2[dangerous] = eps
    g2dd[dangerous] = 0.0

    r2 = g2dd / w2

    lamda = N0 * np.dot(weights, r2).sum() / np.dot(weights, dd)

    wlog = np.exp(np.dot(weights, r2 * np.log(w2) / 2).sum()
        / np.dot(weights, r2).sum())

    return lamda, wlog, Tc(lamda, wlog, mustar)
