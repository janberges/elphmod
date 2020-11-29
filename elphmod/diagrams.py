#/usr/bin/env python

from __future__ import division

import numpy as np

from . import MPI, occupations
comm = MPI.comm
info = MPI.info

def susceptibility(e, kT=0.025, eta=1e-10, occupations=occupations.fermi_dirac):
    r"""Calculate real part of static electronic susceptibility.

    .. math::

        \chi_{\vec q} = \frac 2 N \sum_{\vec k} \frac
            {f(\epsilon_{\vec k + \vec q}) - f(\epsilon_{\vec k})}
            {\epsilon_{\vec k + \vec q} - \epsilon_{\vec k} + \I \eta}

    The resolution in q is limited by the resolution in k.

    Parameters
    ----------
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    kT : float
        Smearing temperature.
    eta : float
        Absolute value of "infinitesimal" imaginary number in denominator.
    occupations : function
        Particle distribution as a function of energy divided by `kT`.

    Returns
    -------
    function
        Static electronic susceptibility as a function of :math:`q_1, q_2 \in
        [0, 2 \pi)`.
    """
    nk, nk = e.shape

    x = e / kT

    f = occupations(x)
    d = occupations.delta(x).sum() / kT

    e = np.tile(e, (2, 2))
    f = np.tile(f, (2, 2))

    scale = nk / (2 * np.pi)
    eta2 = eta ** 2
    prefactor = 2.0 / nk ** 2

    def calculate_susceptibility(q1=0, q2=0):
        q1 = int(round(q1 * scale)) % nk
        q2 = int(round(q2 * scale)) % nk

        if q1 == q2 == 0:
            return -prefactor * d

        df = f[q1:q1 + nk, q2:q2 + nk] - f[:nk, :nk]
        de = e[q1:q1 + nk, q2:q2 + nk] - e[:nk, :nk]

        return prefactor * np.sum(df * de / (de * de + eta2))

    return calculate_susceptibility

def susceptibility2(e, kT=0.025, nmats=1000, hyb_width=1.0, hyb_height=0.0):
    r"""Calculate the Lindhardt bubble using the Green's functions explicitly.

    .. math::

        \chi = \frac \beta 4 - \frac 1 \beta
            \sum_n G(\nu_n) - \frac 1 {(\I \nu_n)^2}

    Only omega = 0 (static) calculation is performed.

    For the treatment of the :math:`1 / \I \nu_n` tail, see:

        Appendix B of the thesis of Hartmut Hafermann.

    Multiply by 2 for spin.

    The resolution in q is limited by the resolution in k.

    Original implementation by Erik G.C.P. van Loon.

    Parameters
    ----------
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    kT : float
        Smearing temperature.
    nmats : int
        Number of fermionic Matsubara frequencies.
    hyb_width : float
        Width of box-shaped hybridization function.
    hyb_height : float
        Height of box-shaped hybridization function.

    Returns
    -------
    function
        Static electronic susceptibility as a function of :math:`q_1, q_2 \in
        [0, 2 \pi)`.
    """
    nk, nk = e.shape

    e = np.tile(e, (2, 2))

    scale = nk / (2 * np.pi)

    prefactor = kT * 4.0 / nk ** 2
    # factor 2 for the negative Matsubara frequencies
    # factor 2 for the spin

    nu = (2 * np.arange(nmats) + 1) * np.pi * kT # Matsubara frequencies

    Delta = -2j * hyb_height * np.arctan(2 * hyb_width / nu) # hybridization

    G = np.empty((nmats, 2 * nk, 2 * nk), dtype=complex) # Green's functions

    for i in range(nmats):
        G[i] = 1.0 / (1j * nu[i] - e - Delta[i])

    tail = -2.0 / (4 * kT) + prefactor * nk ** 2 * np.sum(1.0 / nu ** 2)
    # see Appendix B of the thesis of Hartmut Hafermann
    # factor 2 for spin
    # VERIFY THAT THIS IS CORRECT! (after rewriting function)

    def calculate_susceptibility(q1=0, q2=0):
        q1 = int(round(q1 * scale)) % nk
        q2 = int(round(q2 * scale)) % nk

        Gk  = G[:, :nk, :nk]
        Gkq = G[:, q1:q1 + nk, q2:q2 + nk]

        return prefactor * np.sum(Gk * Gkq) + tail

    return calculate_susceptibility

def polarization(e, c, kT=0.025, eps=1e-15, subspace=None,
        occupations=occupations.fermi_dirac):
    r"""Calculate RPA polarization in orbital basis (density-density).

    .. math::

        \Pi_{\vec q \alpha \beta} = \frac 2 N \sum_{\vec k m n}
            \bracket{\vec k + \vec q m}{\vec k + \vec q \alpha}
            \bracket{\vec k \alpha}{\vec k n}
            \frac
                {f(\epsilon_{\vec k + \vec q m}) - f(\epsilon_{\vec k n})}
                {\epsilon_{\vec k + \vec q m} - \epsilon_{\vec k n}}
            \bracket{\vec k n}{\vec k \beta}
            \bracket{\vec k + \vec q \beta}{\vec k + \vec q m}

    The resolution in q is limited by the resolution in k.

    If `subspace` is given, a cRPA calculation is performed. `subspace` must be
    a boolean array with the same shape as `e`, where ``True`` marks states of
    the target subspace, interactions between which are excluded.

    Parameters
    ----------
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    c : ndarray
        Coefficients for transform to orbital basis. These are given by the
        eigenvectors of the Wannier Hamiltonian.
    kT : float
        Smearing temperature.
    eps : float
        Smallest allowed absolute value of divisor.
    subspace : ndarray or None
        Boolean array to select k points and/or bands in cRPA target subspace.
    occupations : function
        Particle distribution as a function of energy divided by `kT`.

    Returns
    -------
    function
        RPA polarization in orbital basis as a function of :math:`q_1, q_2 \in
        [0, 2 \pi)`.
    """
    cRPA = subspace is not None

    if e.ndim == 2:
        e = e[:, :, np.newaxis]

    if c.ndim == 3:
        c = c[:, :, :, np.newaxis]

    if cRPA and subspace.shape != e.shape:
        subspace = np.reshape(subspace, e.shape)

    nk, nk, nb = e.shape
    nk, nk, no, nb = c.shape # c[k1, k2, a, n] = <k a|k n>

    x = e / kT

    f = occupations(x)
    d = occupations.delta(x) / (-kT)

    e = np.tile(e, (2, 2, 1))
    f = np.tile(f, (2, 2, 1))
    c = np.tile(c, (2, 2, 1, 1))

    if cRPA:
        subspace = np.tile(subspace, (2, 2, 1))

    scale = nk / (2 * np.pi)
    prefactor = 2.0 / nk ** 2

    k1 = slice(0, nk)
    k2 = k1

    dfde = np.empty((nk, nk))

    def calculate_polarization(q1=0, q2=0):
        q1 = int(round(q1 * scale)) % nk
        q2 = int(round(q2 * scale)) % nk

        kq1 = slice(q1, q1 + nk)
        kq2 = slice(q2, q2 + nk)

        Pi = np.empty((nb, nb, no, no), dtype=complex)

        for n in range(nb):
            for m in range(nb):
                df = f[kq1, kq2, m] - f[k1, k2, n]
                de = e[kq1, kq2, m] - e[k1, k2, n]

                ok = abs(de) > eps

                dfde[ ok] = df[ok] / de[ok]
                dfde[~ok] = d[:, :, n][~ok]

                if cRPA:
                    exclude = np.where(
                        subspace[kq1, kq2, m] & subspace[k1, k2, n])

                    dfde[exclude] = 0.0

                cc = c[kq1, kq2, :, m].conj() * c[k1, k2, :, n]

                for a in range(no):
                    cca = cc[:, :, a]

                    for b in range(no):
                        ccb = cc[:, :, b].conj()

                        Pi[n, m, a, b] = np.sum(cca * ccb * dfde)

        return prefactor * Pi.sum(axis=(0, 1))

    return calculate_polarization

def phonon_self_energy(q, e, g2=None, kT=0.025, eps=1e-15,
        occupations=occupations.fermi_dirac, fluctuations=False, Delta=None,
        Delta_diff=False, Delta_occupations=occupations.gauss, Delta_kT=0.025,
        comm=comm):
    r"""Calculate phonon self-energy.

    .. math::

        \Pi_{\vec q \nu} = \frac 2 N \sum_{\vec k m n}
            |g_{\vec q \nu \vec k m n}|^2 \frac
                {f(\epsilon_{\vec k + \vec q m}) - f(\epsilon_{\vec k n})}
                {\epsilon_{\vec k + \vec q m} - \epsilon_{\vec k n}}

    Parameters
    ----------
    q : list of 2-tuples
        Considered q points defined via crystal coordinates :math:`q_1, q_2 \in
        [0, 2 \pi)`.
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    g2 : ndarray
        Squared electron-phonon coupling.
    kT : float
        Smearing temperature.
    eps : float
        Smallest allowed absolute value of divisor.
    occupations : function
        Particle distribution as a function of energy divided by `kT`.
    fluctuations : bool
        Return integrand too (for fluctuation analysis)?
    Delta : float
        Half the width of energy window around Fermi level to be excluded.
    Delta_diff : bool
        Calculate derivative of phonon self-energy w.r.t. `Delta`?
    Delta_occupations : function
        Smoothened Heaviside function to realize excluded energy window.
    Delta_kT : float
        Temperature to smoothen Heaviside function.

    Returns
    -------
    ndarray
        Phonon self-energy.
    """
    nQ = len(q)
    nk = e.shape[0]

    e = np.reshape(e, (nk, nk, -1))
    nbnd = e.shape[-1]

    if g2 is None:
        g2 = np.ones((nQ, 1))

    else:
        g2 = np.reshape(g2, (nQ, -1, nk, nk, nbnd, nbnd))

    nb = g2.shape[1]

    x = e / kT

    f = occupations(x)
    d = occupations.delta(x) / (-kT)

    if Delta is not None:
        x1 = ( e - Delta) / Delta_kT
        x2 = (-e - Delta) / Delta_kT

        Theta = 2 - Delta_occupations(x1) - Delta_occupations(x2)

        if Delta_diff:
            delta = Delta_occupations.delta(x1) + Delta_occupations.delta(x2)
            delta /= -Delta_kT

    e = np.tile(e, (2, 2, 1))
    f = np.tile(f, (2, 2, 1))

    if Delta is not None:
        Theta = np.tile(Theta, (2, 2, 1))

        if Delta_diff:
            delta = np.tile(delta, (2, 2, 1))

    scale = nk / (2 * np.pi)
    prefactor = 2.0 / nk ** 2

    sizes, bounds = MPI.distribute(nQ, bounds=True, comm=comm)

    my_Pi = np.empty((sizes[comm.rank], nb), dtype=g2.dtype)

    if fluctuations:
        my_Pi_k = np.empty((sizes[comm.rank], nb, nk, nk, nbnd, nbnd),
            dtype=g2.dtype)

    dfde = np.empty((nk, nk, nbnd, nbnd))

    k1 = slice(0, nk)
    k2 = slice(0, nk)

    for my_iq, iq in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        q1 = int(round(q[iq, 0] * scale)) % nk
        q2 = int(round(q[iq, 1] * scale)) % nk

        kq1 = slice(q1, q1 + nk)
        kq2 = slice(q2, q2 + nk)

        for m in range(nbnd):
            for n in range(nbnd):
                df = f[kq1, kq2, m] - f[k1, k2, n]
                de = e[kq1, kq2, m] - e[k1, k2, n]

                ok = abs(de) > eps

                dfde[:, :, m, n][ ok] = df[ok] / de[ok]
                dfde[:, :, m, n][~ok] = d[:, :, n][~ok]

                if Delta is not None:
                    if Delta_diff:
                        envelope = (
                              Theta[kq1, kq2, m] * delta[k1, k2, n]
                            + delta[kq1, kq2, m] * Theta[k1, k2, n]
                            )
                    else:
                        envelope = Theta[kq1, kq2, m] * Theta[k1, k2, n]

                    dfde[:, :, m, n] *= envelope

        for nu in range(nb):
            Pi_k = g2[iq, nu] * dfde

            my_Pi[my_iq, nu] = prefactor * Pi_k.sum()

            if fluctuations:
                my_Pi_k[my_iq, nu] = 2 * Pi_k

    Pi = np.empty((nQ, nb), dtype=g2.dtype)

    comm.Allgatherv(my_Pi, (Pi, sizes * nb))

    if fluctuations:
        Pi_k = np.empty((nQ, nb, nk, nk, nbnd, nbnd), dtype=g2.dtype)

        comm.Allgatherv(my_Pi_k, (Pi_k, sizes * nb * nk * nk * nbnd * nbnd))

        return Pi, Pi_k

    else:
        return Pi

def phonon_self_energy2(q, e, g2, kT=0.025, nmats=1000, hyb_width=1.0,
        hyb_height=0.0, GB=4.0):
    """Calculate phonon self-energy using the Green's functions explicitly.

    Parameters
    ----------
    q : list of 2-tuples
        Considered q points defined via crystal coordinates :math:`q_1, q_2 \in
        [0, 2 \pi)`.
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    g2 : ndarray
        Squared electron-phonon coupling.
    kT : float
        Smearing temperature.
    nmats : int
        Number of fermionic Matsubara frequencies.
    hyb_width : float
        Width of box-shaped hybridization function.
    hyb_height : float
        Height of box-shaped hybridization function.
    GB : float
        Memory limit in gigabytes. Exit if exceeded.

    Returns
    -------
    ndarray
        Phonon self-energy.

    See Also
    --------
    susceptibility2 : Similar function with `g2` set to one.
    """
    nk, nk = e.shape
    nQ, nb, nk, nk = g2.shape

    if nmats * (2 * nk) ** 2 * np.dtype(complex).itemsize * comm.size > GB * 1e9:
        info("Error: Memory limit (%g GB) exceeded!" % GB)
        quit()

    e = np.tile(e, (2, 2))

    scale = nk / (2 * np.pi)
    prefactor = kT * 4.0 / nk ** 2

    nu = (2 * np.arange(nmats) + 1) * np.pi * kT # Matsubara frequencies

    Delta = -2j * hyb_height * np.arctan(2 * hyb_width / nu) # hybridization

    G = np.empty((nmats, 2 * nk, 2 * nk), dtype=complex) # Green's functions

    for i in range(nmats):
        G[i] = 1.0 / (1j * nu[i] - e - Delta[i])

    tail = -2.0 / (4 * kT) / nk ** 2 + prefactor * np.sum(1.0 / nu ** 2)
    # VERIFY THAT THIS IS CORRECT!

    sizes, bounds = MPI.distribute(nQ, bounds=True)

    my_Pi = np.empty((sizes[comm.rank], nb), dtype=complex)

    for my_iq, iq in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        q1 = int(round(q[iq, 0] * scale)) % nk
        q2 = int(round(q[iq, 1] * scale)) % nk

        Gk  = G[:, :nk, :nk]
        Gkq = G[:, q1:q1 + nk, q2:q2 + nk]

        chi = prefactor * np.sum(Gk * Gkq, axis=0).real + tail

        for nu in range(nb):
            my_Pi[my_iq, nu] = np.sum(g2[iq, nu] * chi)

    Pi = np.empty((nQ, nb), dtype=complex)

    comm.Allgatherv(my_Pi, (Pi, sizes * nb))

    return Pi

def renormalize_coupling(q, e, g, W, U, nbnd_sub=None, kT=0.025, eps=1e-15,
        occupations=occupations.fermi_dirac, status=True):
    r"""Calculate renormalized electron-phonon coupling.

    .. math::

        \tilde g_{\vec k \vec q i x} = g_{\vec k \vec q i x}
            + \frac 2 N \sum_{\vec k'} g_{\vec k' \vec q i x} \frac
                {f(\epsilon_{\vec k' + \vec q}) - f(\epsilon_{\vec k'})}
                {\epsilon_{\vec k' + \vec q} - \epsilon_{\vec k'}}
            W_{\vec k \vec k' \vec q}

    Parameters
    ----------
    q : list of 2-tuples
        Considered q points defined via crystal coordinates :math:`q_1, q_2 \in
        [0, 2 \pi)`.
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    g : ndarray
        Bare electron-phonon coupling.
    W : ndarray
        Dressed Coulomb interaction.
    U : ndarray
        Eigenvectors of Wannier Hamiltonian belonging to considered band.
    nbnd_sub : int
        Number of bands for Lindhard bubble. Defaults to all bands.
    kT : float
        Smearing temperature.
    eps : float
        Smallest allowed absolute value of divisor.
    occupations : function
        Particle distribution as a function of energy divided by `kT`.
    status : bool
        Print status messages during the calculation?

    Returns
    -------
    ndarray
        Dressed electron-phonon coupling.
    """
    nk, nk, nbnd = e.shape
    nQ, nmodes, nk, nk, nbnd, nbnd = g.shape

    dd = W.ndim == 3

    if dd:
        nQ, norb, norb = W.shape
    else:
        nQ, norb, norb, norb, norb = W.shape

    nk, nk, norb, nbnd = U.shape

    if nbnd_sub is None:
        nbnd_sub = nbnd

    x = e / kT

    f = occupations(x)
    d = occupations.delta(x) / (-kT)

    e = np.tile(e, (2, 2, 1))
    f = np.tile(f, (2, 2, 1))
    U = np.tile(U, (2, 2, 1, 1))

    scale = nk / (2 * np.pi)
    prefactor = 2.0 / nk ** 2

    sizes, bounds = MPI.distribute(nQ, bounds=True)

    my_g_ = np.empty((sizes[comm.rank], nmodes, nk, nk, nbnd, nbnd),
        dtype=complex)

    dfde = np.empty((nk, nk, nbnd_sub, nbnd_sub))

    for my_iq, iq in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        if status:
            print('Renormalize coupling for q point %d..' % (iq + 1))

        q1 = int(round(q[iq, 0] * scale)) % nk
        q2 = int(round(q[iq, 1] * scale)) % nk

        k1 = slice(0, nk)
        k2 = slice(0, nk)

        kq1 = slice(q1, q1 + nk)
        kq2 = slice(q2, q2 + nk)

        for m in range(nbnd_sub):
            for n in range(nbnd_sub):
                df = f[kq1, kq2, m] - f[k1, k2, n]
                de = e[kq1, kq2, m] - e[k1, k2, n]

                ok = abs(de) > eps

                dfde[:, :, m, n][ ok] = df[ok] / de[ok]
                dfde[:, :, m, n][~ok] = d[:, :, n][~ok]

        if dd:
            indices = 'klam,klan,ac,KLcM,KLcN,KLMN,xKLMN->xklmn'
        else:
            indices = 'klam,klbn,bacd,KLcM,KLdN,KLMN,xKLMN->xklmn'

        my_g_[my_iq] = g[iq] + prefactor * np.einsum(indices,
            U[kq1, kq2].conj(), U[k1, k2],
            W[iq],
            U[kq1, kq2, :, :nbnd_sub], U[k1, k2, :, :nbnd_sub].conj(),
            dfde, g[iq, :, :, :, :nbnd_sub, :nbnd_sub])

        #   k+q m           K+q M
        #  ___/___ a     c ___/___
        #     \   \       /   \   \   x q
        #          :::::::         o~~~~~~~
        #  ___\___/       \___\___/
        #     /    b     d    /
        #    k n             K N

    g_ = np.empty((nQ, nmodes, nk, nk, nbnd, nbnd), dtype=complex)

    comm.Allgatherv(my_g_, (g_, sizes * nmodes * nk * nk * nbnd * nbnd))

    return g_

def renormalize_coupling_orbital(W, *args, **kwargs):
    """Calculate renormalized electron-phonon coupling in orbital basis.

    Parameters
    ----------
    W : ndarray
        Dressed Coulomb interaction in orbital basis.
    *args, **kwargs
        Parameters passed to :func:`g_Pi`.

    Returns
    -------
    ndarray
        k-independent change (!) of electron-phonon coupling.

    See Also
    --------
    g_Pi
    """
    dd = W.ndim == 3

    if dd:
        indices = 'qxa,qab->qxb'
    else:
        indices = 'qxab,qabcd->qxcd'

    return np.einsum(indices, g_Pi(*args, dd=dd, **kwargs), W)

def g_Pi(q, e, g, U, kT=0.025, eps=1e-15,
        occupations=occupations.fermi_dirac, dd=True, status=True):
    """Join electron-phonon coupling and Lindhard bubble in orbital basis.

    Parameters
    ----------
    q : list of 2-tuples
        Considered q points defined via crystal coordinates :math:`q_1, q_2 \in
        [0, 2 \pi)`.
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    g : ndarray
        Bare electron-phonon coupling in orbital and displacement basis.
    U : ndarray
        Eigenvectors of Wannier Hamiltonian belonging to considered band.
    kT : float
        Smearing temperature.
    eps : float
        Smallest allowed absolute value of divisor.
    occupations : function
        Particle distribution as a function of energy divided by `kT`.
    dd : bool
        Consider only density-density terms?
    status : bool
        Print status messages during the calculation?

    Returns
    -------
    ndarray
        Product of electron-phonon coupling and Lindhard bubble.
    """
    nk, nk, nbnd = e.shape
    nQ, nmodes, nk, nk, norb, norb = g.shape
    nk, nk, norb, nbnd = U.shape

    x = e / kT

    f = occupations(x)
    d = occupations.delta(x) / (-kT)

    e = np.tile(e, (2, 2, 1))
    f = np.tile(f, (2, 2, 1))
    U = np.tile(U, (2, 2, 1, 1))

    scale = nk / (2 * np.pi)
    prefactor = 2.0 / nk ** 2

    sizes, bounds = MPI.distribute(nQ, bounds=True)

    if dd:
        my_gPi = np.empty((sizes[comm.rank], nmodes, norb), dtype=complex)
    else:
        my_gPi = np.empty((sizes[comm.rank], nmodes, norb, norb), dtype=complex)

    dfde = np.empty((nk, nk, nbnd, nbnd))

    for my_iq, iq in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        if status:
            print('Calculate "g Pi" for q point %d..' % (iq + 1))

        q1 = int(round(q[iq, 0] * scale)) % nk
        q2 = int(round(q[iq, 1] * scale)) % nk

        k1 = slice(0, nk)
        k2 = slice(0, nk)

        kq1 = slice(q1, q1 + nk)
        kq2 = slice(q2, q2 + nk)

        for m in range(nbnd):
            for n in range(nbnd):
                df = f[kq1, kq2, m] - f[k1, k2, n]
                de = e[kq1, kq2, m] - e[k1, k2, n]

                ok = abs(de) > eps

                dfde[:, :, m, n][ ok] = df[ok] / de[ok]
                dfde[:, :, m, n][~ok] = d[:, :, n][~ok]

        if dd:
            indices = 'klcm,klcn,klmn,klam,klbn,xklab->xc'
        else:
            indices = 'klcm,kldn,klmn,klam,klbn,xklab->xcd'

        my_gPi[my_iq] = prefactor * np.einsum(indices,
            U[kq1, kq2], U[k1, k2].conj(), dfde, U[kq1, kq2].conj(), U[k1, k2],
            g[iq])

        #     k+q m
        #  c ___/___ a
        #       \   \   x q
        #            o~~~~~~~
        #    ___\___/
        #  d    /    b
        #      k n

    if dd:
        gPi = np.empty((nQ, nmodes, norb), dtype=complex)
        comm.Allgatherv(my_gPi, (gPi, sizes * nmodes * norb))
    else:
        gPi = np.empty((nQ, nmodes, norb, norb), dtype=complex)
        comm.Allgatherv(my_gPi, (gPi, sizes * nmodes * norb * norb))

    return gPi

def double_fermi_surface_average(q, e, g2, kT=0.025,
        occupations=occupations.fermi_dirac, comm=comm):
    """Calculate double Fermi-surface average.

    Parameters
    ----------
    q : list of 2-tuples
        Considered q points defined via crystal coordinates :math:`q_1, q_2 \in
        [0, 2 \pi)`.
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    g2 : ndarray
        Quantity to be averaged, typically electron-phonon coupling.
    kT : float
        Smearing temperature.
    occupations : function
        Particle distribution as a function of energy divided by `kT`.

    Returns
    -------
    float
        Double Fermi-surface average.
    """
    nk, nk, nel = e.shape
    nQ, nph = g2.shape[:2]

    d = occupations.delta(e / kT) / kT

    e = np.tile(e, (2, 2, 1))
    d = np.tile(d, (2, 2, 1))

    scale = nk / (2 * np.pi)

    sizes, bounds = MPI.distribute(nQ, bounds=True, comm=comm)

    my_av = np.empty((sizes[comm.rank], nph), dtype=g2.dtype)
    my_wg = np.empty(sizes[comm.rank])

    d2 = np.empty((nk, nk, nel, nel))

    k1 = slice(0, nk)
    k2 = slice(0, nk)

    for my_iq, iq in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        q1 = int(round(q[iq, 0] * scale)) % nk
        q2 = int(round(q[iq, 1] * scale)) % nk

        kq1 = slice(q1, q1 + nk)
        kq2 = slice(q2, q2 + nk)

        for m in range(nel):
            for n in range(nel):
                d2[:, :, m, n] = d[kq1, kq2, m] * d[k1, k2, n]

        my_wg[my_iq] = d2.sum()

        for nu in range(nph):
            my_av[my_iq, nu] = (g2[iq, nu] * d2).sum()

    av = np.empty((nQ, nph), dtype=g2.dtype)
    wg = np.empty(nQ)

    comm.Allgatherv(my_av, (av, sizes * nph))
    comm.Allgatherv(my_wg, (wg, sizes))

    return av, wg

def triangle(q1, q2, q3, e, g1, g2, g3, kT=0.025, eps=1e-14,
        occupations=occupations.fermi_dirac, comm=comm):
    r"""Calculate triangle diagram.

    .. math::

        \chi_{\vec q \mu \vec q' \nu}
            &= \sum_{\vec k \alpha \beta \gamma} \frac
            {
                \epsilon_{\vec k \alpha}
                    (f_{\vec k + \vec q \beta} - f_{\vec k + \vec q' \gamma})
                + \epsilon_{\vec k + \vec q \beta}
                    (f_{\vec k + \vec q' \gamma} - f_{\vec k \alpha})
                + \epsilon_{\vec k + \vec q' \gamma}
                    (f_{\vec k \alpha} - f_{\vec k + \vec q \beta})
            }{
                (\epsilon_{\vec k + \vec q \beta}
                    - \epsilon_{\vec k + \vec q' \gamma})
                (\epsilon_{\vec k + \vec q' \gamma}
                    - \epsilon_{\vec k \alpha})
                (\epsilon_{\vec k \alpha}
                    - \epsilon_{\vec k + \vec q \beta})
            }
            \\ &\times
            g^*_{\vec q \mu \vec k \beta \alpha}
            g_{\vec q' \nu \vec k \gamma \alpha}
            g_{\vec q - \vec q' \nu \vec k + \vec q' \beta \gamma}

    .. code-block:: text

                   q u
                    o
                   / \
                  /   \
                 /     \
           k a  v       ^  k+q b
               /         \
              /           \
             /             \
            o------->-------o
        q' v     k+q' c      q-q' v

    Parameters
    ----------
    q1, q2, q3 : 2-tuples
        q points of the vertices in crystal coordinates :math:`q_1, q_2 \in [0,
        2 \pi)`.  In the above diagram, :math:`q_1 = q, q_2 = q', q3 = q - q'`.
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    g1, g2, g3 : ndarray
        Electron-phonon coupling for given q points.
    kT : float
        Smearing temperature.
    eps : float
        Smallest allowed absolute value of divisor.
    occupations : function
        Particle distribution as a function of energy divided by `kT`.

    Returns
    -------
    ndarray
        Value of triangle.
    """
    nk, nk, nbnd = e.shape

    x = e / kT

    f = occupations(x)
    d = occupations.delta(x) / kT

    e  = np.tile(e,  (2, 2, 1))
    f  = np.tile(f,  (2, 2, 1))
    d  = np.tile(d,  (2, 2, 1))
    g1 = np.tile(g1, (2, 2, 1, 1))
    g2 = np.tile(g2, (2, 2, 1, 1))
    g3 = np.tile(g3, (2, 2, 1, 1))

    scale = nk / (2 * np.pi)
    prefactor = 1.0 / nk ** 2

    chi = np.empty((nbnd, nbnd, nbnd, nk, nk), dtype=complex)

    q11 = int(round(q1[0] * scale)) % nk
    q12 = int(round(q1[1] * scale)) % nk

    q21 = int(round(q2[0] * scale)) % nk
    q22 = int(round(q2[1] * scale)) % nk

    k1 = slice(0, nk)
    k2 = slice(0, nk)

    kq1 = slice(q11, q11 + nk)
    kq2 = slice(q12, q12 + nk)

    kQ1 = slice(q21, q21 + nk)
    kQ2 = slice(q22, q22 + nk)

    for a in range(nbnd):
        ea = e[k1, k2, a]
        fa = f[k1, k2, a]
        da = d[k1, k2, a]

        for b in range(nbnd):
            eb = e[kq1, kq2, b]
            fb = f[kq1, kq2, b]
            db = d[kq1, kq2, b]

            for c in range(nbnd):
                ec = e[kQ1, kQ2, c]
                fc = f[kQ1, kQ2, c]
                dc = d[kQ1, kQ2, c]

                dea = eb - ec
                deb = ec - ea
                dec = ea - eb

                dfa = fb - fc
                dfb = fc - fa
                dfc = fa - fb

                la = abs(dea) > eps
                lb = abs(deb) > eps
                lc = abs(dec) > eps

                l = la & lb & lc; L = l
                chi[a, b, c][l] \
                    = (ea[l] * dfa[l] + eb[l] * dfb[l] + ec[l] * dfc[l]) \
                    / (dea[l] * deb[l] * dec[l])

                l = ~la & lb & lc; L |= l
                chi[a, b, c][l] = (db[l] + dfc[l] / dec[l]) / dec[l]

                l = la & ~lb & lc; L |= l
                chi[a, b, c][l] = (dc[l] + dfa[l] / dea[l]) / dea[l]

                l = la & lb & ~lc; L |= l
                chi[a, b, c][l] = (da[l] + dfb[l] / deb[l]) / deb[l]

                l = ~L
                chi[a, b, c][l] = da[l] * (0.5 - fa[l])

    chi = np.einsum('abckl,klba,klca,klbc', chi,
        g1[k1, k2].conj(), g2[k1, k2], g3[kQ1, kQ2])

    return prefactor * chi.sum()
