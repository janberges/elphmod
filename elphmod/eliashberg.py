# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

"""Parameters for McMillan's formula."""

import numpy as np

import elphmod.bravais
import elphmod.diagrams
import elphmod.dos
import elphmod.misc
import elphmod.occupations

def Tc(lamda, wlog, mustar=0.1, w2nd=None, correct=False):
    """Calculate critical temperature using McMillan's formula.

    See Eqs. (2) and (34) by Allen and Dynes, Phys. Rev. B 12, 905 (1975).

    Parameters
    ----------
    lamda : float
        Effective electron-phonon coupling strength.
    wlog : float
        Logarithmic average phonon energy in eV.
    mustar : float
        Coulomb pseudopotential.
    w2nd : bool, default None
        Second-moment average phonon energy used for shape correction in eV.
    correct : bool, default False
        Apply Allen and Dynes' strong-coupling and, if `w2nd` is given, shape
        corrections of Eq. (34)?

    Returns
    -------
    float
        Critical temperature in kelvin.
    """
    Tc = wlog / (1.20 * elphmod.misc.kB) * np.exp(-1.04 * (1 + lamda)
        / max(1e-3, (lamda - mustar * (1 + 0.62 * lamda))))

    if correct:
        # strong-coupling correction:
        Lamda = 2.46 * (1 + 3.8 * mustar)
        Tc *= np.cbrt(1 + (lamda / Lamda) ** 1.5)

        # shape correction:
        if w2nd is not None:
            Lamda = 1.82 * (1 + 6.3 * mustar) * w2nd / wlog
            Tc *= 1 + (w2nd / wlog - 1) * lamda ** 2 / (lamda ** 2 + Lamda ** 2)

    return Tc

def McMillan(nq, e, w2, g2, eps=1e-10, mustar=0.0, tetra=False, kT=0.025,
        f='fd', correct=False):
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
        couplings are set to zero.
    mustar : float
        Coulomb pseudopotential.
    tetra : bool
        Calculate double Fermi-surface average and density of states using 2D
        tetrahedron methods? Otherwise summations over broadened delta functions
        are performed.
    kT : float
        Smearing temperature. Only used if `tetra` is ``False``.
    f : function
        Particle distribution as a function of energy divided by `kT`. Only used
        if `tetra` is ``False``.
    correct : bool, default False
        Apply Allen and Dynes' strong-coupling and shape corrections?

    Returns
    -------
    float
        Effective electron-phonon coupling strength :math:`\lambda`.
    float
        Effective phonon frequency :math:`\langle \omega \rangle`.
    float
        Estimated critical temperature of superconductivity.
    float, optional
        Second-moment average phonon energy used for shape correction.
    """
    f = elphmod.occupations.smearing(f)

    nk, nk, nel = e.shape
    nQ, nph = w2.shape
    nQ, nph, nk, nk, nel, nel = g2.shape

    q = np.array(sorted(elphmod.bravais.irreducibles(nq)))

    weights = np.array([len(elphmod.bravais.images(q1, q2, nq))
        for q1, q2 in q])

    if tetra:
        q *= nk // nq

        g2dd = np.zeros((nQ, nph))
        dd = np.zeros(nQ)

        for iq, (q1, q2) in enumerate(q):
            E = np.roll(np.roll(e, shift=-q1, axis=0), shift=-q2, axis=1)

            g2_fun = elphmod.bravais.linear_interpolation(g2[iq], axes=(1, 2))

            for n in range(nel):
                for m in range(nel):
                    intersections = elphmod.dos.double_delta(e[:, :, n],
                        E[:, :, m])(0)

                    for (k1, k2), weight in intersections.items():
                        g2dd[iq] += weight * g2_fun(k1, k2)[:, n, m]
                        dd[iq] += weight

        N0 = 0

        for n in range(nel):
            N0 += elphmod.dos.hexDOS(e[:, :, n])(0)
    else:
        q = 2 * np.pi / nq * q

        g2dd, dd = elphmod.diagrams.double_fermi_surface_average(q,
            e, g2, kT, f)

        N0 = f.delta(e / kT).sum() / kT / np.prod(e.shape[:-1])

    g2dd *= weights[:, np.newaxis]
    dd *= weights

    w2 = w2.copy()

    dangerous = np.where(w2 < eps)

    w2[dangerous] = eps
    g2dd[dangerous] = 0.0

    V = g2dd / w2

    lamda = N0 * V.sum() / dd.sum()
    wlog = np.exp((V * np.log(w2) / 2).sum() / V.sum())

    if correct:
        w2nd = np.sqrt((V * w2).sum() / V.sum())

        return lamda, wlog, Tc(lamda, wlog, mustar, w2nd, correct=True), w2nd

    return lamda, wlog, Tc(lamda, wlog, mustar)
