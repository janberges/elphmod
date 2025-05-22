# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

"""Susceptibilities, self-energies, etc."""

import numpy as np

import elphmod.misc
import elphmod.MPI
import elphmod.occupations

comm = elphmod.MPI.comm
info = elphmod.MPI.info

def susceptibility(e, kT=0.025, eta=1e-10, occupations='fd'):
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
    occupations = elphmod.occupations.smearing(occupations)

    nk, nk = e.shape

    x = e / kT

    f = occupations(x)
    d = occupations.delta(x).sum() / kT

    e = np.tile(e, (2, 2))
    f = np.tile(f, (2, 2))

    scale = nk / (2 * np.pi)
    eta2 = eta ** 2
    prefactor = 2 / nk ** 2

    def calculate_susceptibility(q1=0, q2=0):
        q1 = int(round(q1 * scale)) % nk
        q2 = int(round(q2 * scale)) % nk

        if q1 == q2 == 0:
            return prefactor * -d

        df = f[q1:q1 + nk, q2:q2 + nk] - f[:nk, :nk]
        de = e[q1:q1 + nk, q2:q2 + nk] - e[:nk, :nk]

        return prefactor * np.sum(df * de / (de * de + eta2))

    return calculate_susceptibility

def susceptibility2(e, kT=0.025, nmats=1000, hyb_width=1.0, hyb_height=0.0):
    r"""Calculate the Lindhard bubble using the Green's functions explicitly.

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

    prefactor = 4 * kT / nk ** 2
    # factor 2 for the negative Matsubara frequencies
    # factor 2 for the spin

    nu = (2 * np.arange(nmats) + 1) * np.pi * kT # Matsubara frequencies

    Delta = -2j * hyb_height * np.arctan(2 * hyb_width / nu) # hybridization

    G = np.empty((nmats, 2 * nk, 2 * nk), dtype=complex) # Green's functions

    for i in range(nmats):
        G[i] = 1 / (1j * nu[i] - e - Delta[i])

    tail = -2 / (4 * kT) + prefactor * nk ** 2 * np.sum(1 / nu ** 2)
    # see Appendix B of the thesis of Hartmut Hafermann
    # factor 2 for spin
    # VERIFY THAT THIS IS CORRECT! (after rewriting function)

    def calculate_susceptibility(q1=0, q2=0):
        q1 = int(round(q1 * scale)) % nk
        q2 = int(round(q2 * scale)) % nk

        Gk = G[:, :nk, :nk]
        Gkq = G[:, q1:q1 + nk, q2:q2 + nk]

        return prefactor * np.sum(Gk * Gkq) + tail

    return calculate_susceptibility

def polarization(e, U, kT=0.025, eps=1e-10, subspace=None, occupations='fd'):
    r"""Calculate RPA polarization in orbital basis (density-density).

    .. math::

        \Pi_{\vec q \alpha \beta} = \frac 2 N \sum_{\vec k m n}
            \bracket{\vec k + \vec q \alpha}{\vec k + \vec q m}
            \bracket{\vec k n}{\vec k \alpha}
            \frac
                {f(\epsilon_{\vec k + \vec q m}) - f(\epsilon_{\vec k n})}
                {\epsilon_{\vec k + \vec q m} - \epsilon_{\vec k n}}
            \bracket{\vec k + \vec q m}{\vec k + \vec q \beta}
            \bracket{\vec k \beta}{\vec k n}

    The resolution in q is limited by the resolution in k.

    If `subspace` is given, a cRPA calculation is performed. `subspace` must be
    a boolean array with the same shape as `e`, where ``True`` marks states of
    the target subspace, interactions between which are excluded.

    Parameters
    ----------
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    U : ndarray
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
        RPA polarization in orbital basis as a function of :math:`q_1, q_2, q_3
        \in [0, 2 \pi)`.
    """
    occupations = elphmod.occupations.smearing(occupations)

    cRPA = subspace is not None

    nk_orig = e.shape[:-1]
    nk = np.ones(3, dtype=int)
    nk[:len(nk_orig)] = nk_orig

    nbnd = e.shape[-1]
    e = np.reshape(e, (*nk, nbnd))
    U = np.reshape(U, (*nk, -1, nbnd))
    # U[k1, k2, k3, a, n] = <k a|k n>
    norb = U.shape[3]

    if cRPA and subspace.shape != e.shape:
        subspace = np.reshape(subspace, e.shape)

    x = e / kT

    f = occupations(x)
    d = occupations.delta(x) / (-kT)

    e = np.tile(e, (2, 2, 2, 1))
    f = np.tile(f, (2, 2, 2, 1))
    U = np.tile(U, (2, 2, 2, 1, 1))

    if cRPA:
        subspace = np.tile(subspace, (2, 2, 2, 1))

    scale = nk / (2 * np.pi)
    prefactor = 2 / nk.prod()

    k1 = slice(0, nk[0])
    k2 = slice(0, nk[1])
    k3 = slice(0, nk[2])

    dfde = np.empty(tuple(nk))

    def calculate_polarization(q1=0, q2=0, q3=0):
        q = np.array([q1, q2, q3])

        q1, q2, q3 = np.round(q * scale).astype(int) % nk

        kq1 = slice(q1, q1 + nk[0])
        kq2 = slice(q2, q2 + nk[1])
        kq3 = slice(q3, q3 + nk[2])

        Pi = np.empty((nbnd, nbnd, norb, norb), dtype=complex)

        for m in range(nbnd):
            for n in range(nbnd):
                df = f[kq1, kq2, kq3, m] - f[k1, k2, k3, n]
                de = e[kq1, kq2, kq3, m] - e[k1, k2, k3, n]

                ok = abs(de) > eps

                dfde[ok] = df[ok] / de[ok]
                dfde[~ok] = d[..., n][~ok]

                if cRPA:
                    exclude = np.where(
                        subspace[kq1, kq2, kq3, m] & subspace[k1, k2, k3, n])

                    dfde[exclude] = 0.0

                UU = U[kq1, kq2, kq3, :, m].conj() * U[k1, k2, k3, :, n]

                Pi[m, n] = np.einsum('ijka,ijk,ijkb->ab', UU.conj(), dfde, UU)

        return prefactor * Pi.sum(axis=(0, 1))

    return calculate_polarization

def phonon_self_energy(q, e, g2=None, kT=0.025, eps=1e-10, omega=0.0,
        occupations='fd', fluctuations=False, Delta=None, Delta_diff=False,
        Delta_occupations='gauss', Delta_kT=0.025, g=None, G=None,
        symmetrize=True, diagonal=False, U=0.0, comm=comm):
    r"""Calculate phonon self-energy.

    .. math::

        \Pi_{\vec q \nu}(\omega) = \frac 2 N \sum_{\vec k m n}
            |g_{\vec q \nu \vec k m n}|^2 \frac
                {f(\epsilon_{\vec k n}) - f(\epsilon_{\vec k + \vec q m})}
                {\epsilon_{\vec k n} - \epsilon_{\vec k + \vec q m} + \omega}

    Parameters
    ----------
    q : list of tuple
        List of q points in crystal coordinates :math:`q_i \in [0, 2 \pi)`.
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    g2 : ndarray
        Squared electron-phonon coupling. The resulting phonon self-energy will
        be a vector in the space of phonon modes.
    kT : float
        Smearing temperature.
    eps : float
        Smallest allowed absolute value of divisor.
    omega : float
        Nonadiabatic frequency argument; shall include small imaginary
        regulator if nonzero.
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
    g : ndarray
        Electron-phonon coupling on one side of the bubble. If given, `g2` is
        ignored. The resulting phonon self-energy will be a matrix (an outer
        product of `g` and `G`) in the space of phonon modes or displacements.
    G : ndarray
        Electron-phonon coupling on the other side of the bubble. If absent, `g`
        is used.
    symmetrize : bool
        Symmetrize phonon self-energy with respect to swapping `g` and `G`?
    diagonal : bool
        Neglect off-diagonal elements when using `g` and `G`?
    U : ndarray
        Contact interaction (a matrix in band indices) to model the effect of
        excitons. Only used if `g` is present. Associated terms are currently
        excluded from the integrand provided for fluctuation diagnostics.

    Returns
    -------
    ndarray
        Phonon self-energy.
    """
    occupations = elphmod.occupations.smearing(occupations)
    Delta_occupations = elphmod.occupations.smearing(Delta_occupations)

    nQ = len(q)

    q_orig = q
    q = np.zeros((nQ, 3))
    q[:, :len(q_orig[0])] = q_orig

    nk_orig = e.shape[:-1]
    nk = np.ones(3, dtype=int)
    nk[:len(nk_orig)] = nk_orig

    nbnd = e.shape[-1]
    e = np.reshape(e, (*nk, nbnd))

    if g is None:
        if g2 is None:
            g2 = np.ones((nQ, 1))
        else:
            g2 = np.reshape(g2, (nQ, -1, *nk, nbnd, nbnd))

        phshape = g2.shape[1:2]
    else:
        g = np.reshape(g, (nQ, -1, *nk, nbnd, nbnd))

        if G is None:
            G = g
        else:
            G = np.reshape(G, g.shape)

        phshape = g.shape[1:2] * 2

    x = e / kT

    f = occupations(x)
    d = occupations.delta(x) / (-kT)

    if Delta is not None:
        x1 = (e - Delta) / Delta_kT
        x2 = (-e - Delta) / Delta_kT

        Theta = 2 - Delta_occupations(x1) - Delta_occupations(x2)

        if Delta_diff:
            delta = Delta_occupations.delta(x1) + Delta_occupations.delta(x2)
            delta /= -Delta_kT

    if np.any(q != 0):
        e = np.tile(e, (2, 2, 2, 1))
        f = np.tile(f, (2, 2, 2, 1))

        if Delta is not None:
            Theta = np.tile(Theta, (2, 2, 2, 1))

            if Delta_diff:
                delta = np.tile(delta, (2, 2, 2, 1))

    scale = nk / (2 * np.pi)
    prefactor = 2 / nk.prod()

    sizes, bounds = elphmod.MPI.distribute(nQ, bounds=True, comm=comm)

    omega = np.array(omega)

    U = np.array(U)

    my_Pi = np.empty((sizes[comm.rank], *phshape, *omega.shape),
        dtype=float if np.isrealobj(omega) and np.isrealobj(g2)
            and np.isrealobj(g) and np.isrealobj(G) else complex)

    if fluctuations:
        my_Pi_k = np.empty((sizes[comm.rank], *phshape, *omega.shape,
            *nk, nbnd, nbnd), dtype=my_Pi.dtype)

    dfde = np.empty((*omega.shape, *nk, nbnd, nbnd),
        dtype=float if np.isrealobj(omega) else complex)

    k1 = slice(0, nk[0])
    k2 = slice(0, nk[1])
    k3 = slice(0, nk[2])

    dynamic = np.any(omega != 0)

    status = elphmod.misc.StatusBar(sizes[comm.rank] * np.prod(phshape),
        title='calculate %s phonon self-energy'
            % ('dynamic' if dynamic else 'static'))

    for my_iq, iq in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        q1, q2, q3 = np.round(q[iq] * scale).astype(int) % nk

        kq1 = slice(q1, q1 + nk[0])
        kq2 = slice(q2, q2 + nk[1])
        kq3 = slice(q3, q3 + nk[2])

        for m in range(nbnd):
            for n in range(nbnd):
                df = f[k1, k2, k3, n] - f[kq1, kq2, kq3, m]
                de = e[k1, k2, k3, n] - e[kq1, kq2, kq3, m]

                if dynamic:
                    dfde[..., m, n] = df / np.add.outer(omega, de)
                else:
                    ok = abs(de) > eps

                    dfde[..., m, n][ok] = df[ok] / de[ok]
                    dfde[..., m, n][~ok] = d[..., n][~ok]

                if Delta is not None:
                    if Delta_diff:
                        envelope = (
                              Theta[kq1, kq2, kq3, m] * delta[k1, k2, k3, n]
                            + delta[kq1, kq2, kq3, m] * Theta[k1, k2, k3, n]
                        )
                    else:
                        envelope = (Theta[kq1, kq2, kq3, m]
                            * Theta[k1, k2, k3, n])

                    dfde[..., m, n] *= envelope

        for nu in np.ndindex(*phshape):
            if g is None:
                Pi_k = g2[iq][nu] * dfde
            elif diagonal and nu[0] != nu[1]:
                Pi_k = 0 * dfde
            else:
                G2 = g[iq, nu[0]].conj() * G[iq, nu[1]]

                if G is not g and symmetrize:
                    G2 += G[iq, nu[0]].conj() * g[iq, nu[1]]
                    G2 /= 2

                Pi_k = G2 * dfde

            my_Pi[my_iq][nu] = prefactor * Pi_k.sum(axis=tuple(range(-5, 0)))

            if fluctuations:
                my_Pi_k[my_iq][nu] = 2 * Pi_k

            status.update()

        if np.any(U != 0):
            axes = tuple(range(-5, -2))

            chi0 = prefactor * dfde.sum(axis=axes)

            W = U / (1 - chi0 * U)

            PiL = [prefactor * (g[iq, nu].conj() * dfde).sum(axis=axes)
                for nu in range(g.shape[1])]

            PiR = [prefactor * (dfde * G[iq, nu]).sum(axis=axes)
                for nu in range(G.shape[1])]

            my_Pi[my_iq] += np.einsum('u...mn,...mn,v...mn->uv...', PiL, W, PiR)

    Pi = np.empty((nQ, *phshape, *omega.shape), dtype=my_Pi.dtype)

    comm.Allgatherv(my_Pi, (Pi, comm.allgather(my_Pi.size)))

    if fluctuations:
        Pi_k = np.empty((nQ, *phshape, *omega.shape, *nk_orig, nbnd, nbnd),
            dtype=my_Pi_k.dtype)

        comm.Allgatherv(my_Pi_k, (Pi_k, comm.allgather(my_Pi_k.size)))

        return Pi, Pi_k
    else:
        return Pi

def phonon_self_energy_fermi_shift(e, g, kT=0.025, occupations='fd'):
    r"""Calculate phonon self-energy arising from change of chemical potential.

    :func:`phonon_self_energy` provides the second derivative of the grand
    potential. To obtain the second derivative of the free energy, this shift
    associated with a possible change of the chemical potential has to be added.

    Parameters
    ----------
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    g : ndarray
        Electron-phonon coupling for :math:`\vec q = 0`.
    kT : float
        Smearing temperature.
    occupations : function
        Particle distribution as a function of energy divided by `kT`.

    Returns
    -------
    ndarray
        Phonon self-energy correction.
    """
    occupations = elphmod.occupations.smearing(occupations)

    nk = np.ones(3, dtype=int)
    nk[:len(e.shape[:-1])] = e.shape[:-1]

    nbnd = e.shape[-1]
    e = np.reshape(e, (*nk, nbnd))
    g = np.reshape(g, (-1, *nk, nbnd, nbnd))

    x = e / kT
    d = occupations.delta(x) / kT

    dos = d.sum()
    avg = np.einsum('ijkn,xijknn->x', d, g) / dos

    return 2 / nk.prod() * dos * np.outer(avg, avg) # one complex conjugate?

def phonon_self_energy2(q, e, g2, kT=0.025, nmats=1000, hyb_width=1.0,
        hyb_height=0.0, GB=4.0):
    r"""Calculate phonon self-energy using the Green's functions explicitly.

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
    nQ, nmodes, nk, nk = g2.shape

    if (nmats * (2 * nk) ** 2 * np.dtype(complex).itemsize * comm.size
            > GB * 1e9):
        info('Memory limit (%g GB) exceeded!' % GB, error=True)

    e = np.tile(e, (2, 2))

    scale = nk / (2 * np.pi)
    prefactor = kT * 4 / nk ** 2

    nu = (2 * np.arange(nmats) + 1) * np.pi * kT # Matsubara frequencies

    Delta = -2j * hyb_height * np.arctan(2 * hyb_width / nu) # hybridization

    G = np.empty((nmats, 2 * nk, 2 * nk), dtype=complex) # Green's functions

    for i in range(nmats):
        G[i] = 1 / (1j * nu[i] - e - Delta[i])

    tail = -2 / (4 * kT) / nk ** 2 + prefactor * np.sum(1 / nu ** 2)
    # VERIFY THAT THIS IS CORRECT!

    sizes, bounds = elphmod.MPI.distribute(nQ, bounds=True)

    my_Pi = np.empty((sizes[comm.rank], nmodes), dtype=complex)

    for my_iq, iq in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        q1 = int(round(q[iq, 0] * scale)) % nk
        q2 = int(round(q[iq, 1] * scale)) % nk

        Gk = G[:, :nk, :nk]
        Gkq = G[:, q1:q1 + nk, q2:q2 + nk]

        chi = prefactor * np.sum(Gk * Gkq, axis=0).real + tail

        for nu in range(nmodes):
            my_Pi[my_iq, nu] = np.sum(g2[iq, nu] * chi)

    Pi = np.empty((nQ, nmodes), dtype=complex)

    comm.Allgatherv(my_Pi, (Pi, sizes * nmodes))

    return Pi

def renormalize_coupling_band(q, e, g, W, U, kT=0.025, eps=1e-10,
        occupations='fd', nbnd_sub=None, status=True):
    r"""Calculate renormalized electron-phonon coupling in band basis.

    .. math::

        \tilde g_{\vec q \nu \vec k m n} = g_{\vec q \nu \vec k m n}
            + \frac 2 N \sum_{\vec k' m' n' \alpha \beta \gamma \delta}
            U_{\vec k + \vec q \alpha m}^* U_{\vec k \beta n}
            W_{\vec q \alpha \beta \gamma \delta}
            U_{\vec k' + \vec q \gamma m'} U_{\vec k' \delta n'}^*
            \frac
                {f(\epsilon_{\vec k' + \vec q m'}) - f(\epsilon_{\vec k' n'})}
                {\epsilon_{\vec k' + \vec q m'} - \epsilon_{\vec k' n'}}
            g_{\vec q \nu \vec k' m' n'}

    Parameters
    ----------
    q : list of tuple
        List of q points in crystal coordinates :math:`q_i \in [0, 2 \pi)`.
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    g : ndarray
        Bare electron-phonon coupling in band basis.
    W : ndarray
        Dressed q-dependent Coulomb interaction in orbital basis.
    U : ndarray
        Eigenvectors of Wannier Hamiltonian belonging to considered bands.
    kT : float
        Smearing temperature.
    eps : float
        Smallest allowed absolute value of divisor.
    occupations : function
        Particle distribution as a function of energy divided by `kT`.
    nbnd_sub : int
        Number of bands for Lindhard bubble. Defaults to all bands.
    status : bool
        Print status messages during the calculation?

    Returns
    -------
    ndarray
        Dressed electron-phonon coupling in band basis.

    See Also
    --------
    renormalize_coupling_orbital
    """
    occupations = elphmod.occupations.smearing(occupations)

    nQ = len(q)

    q_orig = q
    q = np.zeros((nQ, 3))
    q[:, :len(q_orig[0])] = q_orig

    nk_orig = e.shape[:-1]
    nk = np.ones(3, dtype=int)
    nk[:len(nk_orig)] = nk_orig

    nbnd = e.shape[-1]
    e = np.reshape(e, (*nk, nbnd))

    U = np.reshape(U, (*nk, -1, nbnd))
    norb = U.shape[3]

    g = np.reshape(g, (nQ, -1, *nk, nbnd, nbnd))
    nmodes = g.shape[1]

    dd = W.ndim == 3

    if dd:
        W = np.reshape(W, (nQ, norb, norb))
    else:
        W = np.reshape(W, (nQ, norb, norb, norb, norb))

    if nbnd_sub is None:
        nbnd_sub = nbnd

    x = e / kT

    f = occupations(x)
    d = occupations.delta(x) / (-kT)

    e = np.tile(e, (2, 2, 2, 1))
    f = np.tile(f, (2, 2, 2, 1))
    U = np.tile(U, (2, 2, 2, 1, 1))

    scale = nk / (2 * np.pi)
    prefactor = 2 / nk.prod()

    sizes, bounds = elphmod.MPI.distribute(nQ, bounds=True)

    my_g_ = np.empty((sizes[comm.rank], nmodes, *nk, nbnd, nbnd), dtype=complex)

    dfde = np.empty((*nk, nbnd_sub, nbnd_sub))

    for my_iq, iq in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        if status:
            print('Renormalize coupling for q point %d' % (iq + 1))

        q1, q2, q3 = np.round(q[iq] * scale).astype(int) % nk

        k1 = slice(0, nk[0])
        k2 = slice(0, nk[1])
        k3 = slice(0, nk[2])

        kq1 = slice(q1, q1 + nk[0])
        kq2 = slice(q2, q2 + nk[1])
        kq3 = slice(q3, q3 + nk[2])

        for m in range(nbnd_sub):
            for n in range(nbnd_sub):
                df = f[kq1, kq2, kq3, m] - f[k1, k2, k3, n]
                de = e[kq1, kq2, kq3, m] - e[k1, k2, k3, n]

                ok = abs(de) > eps

                dfde[..., m, n][ok] = df[ok] / de[ok]
                dfde[..., m, n][~ok] = d[..., n][~ok]

        if dd:
            indices = 'IJKcM,IJKcN,IJKMN,xIJKMN->xc'
        else:
            indices = 'IJKcM,IJKdN,IJKMN,xIJKMN->xcd'

        Pig = prefactor * np.einsum(indices,
            U[kq1, kq2, kq3, :, :nbnd_sub], U[k1, k2, k3, :, :nbnd_sub].conj(),
            dfde, g[iq, :, k1, k2, k3, :nbnd_sub, :nbnd_sub])

        if dd:
            indices = 'ijkam,ijkan,ac,xc->xijkmn'
        else:
            indices = 'ijkam,ijkbn,abcd,xcd->xijkmn'

        my_g_[my_iq] = g[iq] + np.einsum(indices,
            U[kq1, kq2, kq3].conj(), U[k1, k2, k3], W[iq], Pig)

        #   k+q m           K+q M
        #  ___/___ a     c ___/___
        #     \   \       /   \   \   x q
        #          :::::::         o~~~~~~~
        #  ___\___/       \___\___/
        #     /    b     d    /
        #    k n             K N

    g_ = np.empty((nQ, nmodes, *nk_orig, nbnd, nbnd), dtype=complex)

    comm.Allgatherv(my_g_, (g_, sizes * nmodes * nk.prod() * nbnd * nbnd))

    return g_

def renormalize_coupling_orbital(q, e, g, W, U, **kwargs):
    r"""Calculate renormalized electron-phonon coupling in orbital basis.

    .. math::

        \tilde g_{\vec q \nu \vec k \alpha \beta}
            = g_{\vec q \nu \vec k \alpha \beta}
            + \frac 2 N \sum_{\vec k m n \alpha' \beta' \gamma \delta}
            W_{\vec q \alpha \beta \gamma \delta}
            U_{\vec k + \vec q \gamma m} U_{\vec k \delta n}^*
            \frac
                {f(\epsilon_{\vec k + \vec q m}) - f(\epsilon_{\vec k n})}
                {\epsilon_{\vec k + \vec q m} - \epsilon_{\vec k n}}
            U_{\vec k + \vec q \alpha' m}^* U_{\vec k \beta' n}
            g_{\vec q \nu \vec k \alpha' \beta'}

    Parameters
    ----------
    q : list of tuple
        List of q points in crystal coordinates :math:`q_i \in [0, 2 \pi)`.
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    g : ndarray
        Bare electron-phonon coupling in orbital basis.
    W : ndarray
        Dressed q-dependent Coulomb interaction in orbital basis.
    U : ndarray
        Eigenvectors of Wannier Hamiltonian belonging to considered bands.
    **kwargs
        Parameters passed to :func:`Pi_g`.

    Returns
    -------
    ndarray
        Dressed electron-phonon coupling in orbital basis.

    See Also
    --------
    Pi_g
    renormalize_coupling_orbital
    """
    dd = W.ndim == 3

    if dd:
        indices = 'qac,qxc->qxa'
    else:
        indices = 'qabcd,qxcd->qxab'

    dg = np.einsum(indices, W, Pi_g(q, e, g, U, dd=dd, **kwargs))

    #            k+q m
    #   a     c ___/___ a'
    #  \   q   /   \   \   x q
    #   :::::::         o~~~~~~~
    #  /       \___\___/
    #   b     d    /    b'
    #             k n

    g = g.copy()

    while dg.ndim < g.ndim - 1 if dd else g.ndim:
        dg = dg[:, :, np.newaxis]

    if dd:
        for a in range(dg.shape[-1]):
            g[..., a, a] += dg[..., a]
    else:
        g += dg

    return g

def Pi_g(q, e, g, U, kT=0.025, eps=1e-10, occupations='fd', dd=True,
        status=True):
    r"""Join electron-phonon coupling and Lindhard bubble in orbital basis.

    Parameters
    ----------
    q : list of tuple
        List of q points in crystal coordinates :math:`q_i \in [0, 2 \pi)`.
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    g : ndarray
        Bare electron-phonon coupling in orbital basis.
    U : ndarray
        Eigenvectors of Wannier Hamiltonian belonging to considered bands.
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
        (k-independent) product of electron-phonon coupling and Lindhard bubble.
    """
    occupations = elphmod.occupations.smearing(occupations)

    nQ = len(q)

    q_orig = q
    q = np.zeros((nQ, 3))
    q[:, :len(q_orig[0])] = q_orig

    nk_orig = e.shape[:-1]
    nk = np.ones(3, dtype=int)
    nk[:len(nk_orig)] = nk_orig

    nbnd = e.shape[-1]
    e = np.reshape(e, (*nk, nbnd))

    U = np.reshape(U, (*nk, -1, nbnd))
    norb = U.shape[3]

    g = np.reshape(g, (nQ, -1, *nk, norb, norb))
    nmodes = g.shape[1]

    x = e / kT

    f = occupations(x)
    d = occupations.delta(x) / (-kT)

    e = np.tile(e, (2, 2, 2, 1))
    f = np.tile(f, (2, 2, 2, 1))
    U = np.tile(U, (2, 2, 2, 1, 1))

    scale = nk / (2 * np.pi)
    prefactor = 2 / nk.prod()

    sizes, bounds = elphmod.MPI.distribute(nQ, bounds=True)

    if dd:
        my_Pig = np.empty((sizes[comm.rank], nmodes, norb), dtype=complex)
    else:
        my_Pig = np.empty((sizes[comm.rank], nmodes, norb, norb), dtype=complex)

    dfde = np.empty((*nk, nbnd, nbnd))

    for my_iq, iq in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        if status:
            print('Calculate "g Pi" for q point %d' % (iq + 1))

        q1, q2, q3 = np.round(q[iq] * scale).astype(int) % nk

        k1 = slice(0, nk[0])
        k2 = slice(0, nk[1])
        k3 = slice(0, nk[2])

        kq1 = slice(q1, q1 + nk[0])
        kq2 = slice(q2, q2 + nk[1])
        kq3 = slice(q3, q3 + nk[2])

        for m in range(nbnd):
            for n in range(nbnd):
                df = f[kq1, kq2, kq3, m] - f[k1, k2, k3, n]
                de = e[kq1, kq2, kq3, m] - e[k1, k2, k3, n]

                ok = abs(de) > eps

                dfde[..., m, n][ok] = df[ok] / de[ok]
                dfde[..., m, n][~ok] = d[..., n][~ok]

        if dd:
            indices = 'ijkcm,ijkcn,ijkmn,ijkam,ijkbn,xijkab->xc'
        else:
            indices = 'ijkcm,ijkdn,ijkmn,ijkam,ijkbn,xijkab->xcd'

        my_Pig[my_iq] = prefactor * np.einsum(indices,
            U[kq1, kq2, kq3], U[k1, k2, k3].conj(), dfde,
            U[kq1, kq2, kq3].conj(), U[k1, k2, k3], g[iq])

        #     k+q m
        #  c ___/___ a
        #       \   \   x q
        #            o~~~~~~~
        #    ___\___/
        #  d    /    b
        #      k n

    if dd:
        Pig = np.empty((nQ, nmodes, norb), dtype=complex)
        comm.Allgatherv(my_Pig, (Pig, sizes * nmodes * norb))
    else:
        Pig = np.empty((nQ, nmodes, norb, norb), dtype=complex)
        comm.Allgatherv(my_Pig, (Pig, sizes * nmodes * norb * norb))

    return Pig

def double_fermi_surface_average(q, e, g2=None, kT=0.025, occupations='fd',
        comm=comm):
    r"""Calculate double Fermi-surface average.

    Please note that not the average itself is returned!

    .. math::

        \langle g^2 \rangle = \frac {
            \sum_{\vec q \nu \vec k m n}
            |g_{\vec q \nu \vec k m n}|^2
            \delta(\epsilon_{\vec k n})
            \delta(\epsilon_{\vec k + \vec q m})
        }{
            \sum_{\vec q \vec k m n}
            \delta(\epsilon_{\vec k n})
            \delta(\epsilon_{\vec k + \vec q m})
        }

    Parameters
    ----------
    q : list of tuple
        List of q points in crystal coordinates :math:`q_i \in [0, 2 \pi)`.
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    g2 : ndarray
        Quantity to be averaged, typically squared electron-phonon coupling.
    kT : float
        Smearing temperature.
    occupations : function
        Particle distribution as a function of energy divided by `kT`.

    Returns
    -------
    ndarray
        Enumerator of double Fermi-surface average before :math:`\vec q \nu`
        summation.
    ndarray
        Denominator of double Fermi-surface average before :math:`\vec q`
        summation.
    """
    occupations = elphmod.occupations.smearing(occupations)

    nQ = len(q)

    q_orig = q
    q = np.zeros((nQ, 3))
    q[:, :len(q_orig[0])] = q_orig

    nk_orig = e.shape[:-1]
    nk = np.ones(3, dtype=int)
    nk[:len(nk_orig)] = nk_orig

    nbnd = e.shape[-1]
    e = np.reshape(e, (*nk, nbnd))

    if g2 is None:
        g2 = np.ones((nQ, 1))

    else:
        g2 = np.reshape(g2, (nQ, -1, *nk, nbnd, nbnd))

    nmodes = g2.shape[1]

    d = occupations.delta(e / kT) / kT

    e = np.tile(e, (2, 2, 2, 1))
    d = np.tile(d, (2, 2, 2, 1))

    scale = nk / (2 * np.pi)

    sizes, bounds = elphmod.MPI.distribute(nQ, bounds=True, comm=comm)

    my_enum = np.empty((sizes[comm.rank], nmodes),
        dtype=float if np.isrealobj(g2) else complex)
    my_deno = np.empty(sizes[comm.rank])

    d2 = np.empty((*nk, nbnd, nbnd))

    k1 = slice(0, nk[0])
    k2 = slice(0, nk[1])
    k3 = slice(0, nk[2])

    for my_iq, iq in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        q1, q2, q3 = np.round(q[iq] * scale).astype(int) % nk

        kq1 = slice(q1, q1 + nk[0])
        kq2 = slice(q2, q2 + nk[1])
        kq3 = slice(q3, q3 + nk[2])

        for m in range(nbnd):
            for n in range(nbnd):
                d2[..., m, n] = d[kq1, kq2, kq3, m] * d[k1, k2, k3, n]

        for nu in range(nmodes):
            my_enum[my_iq, nu] = (g2[iq, nu] * d2).sum()

        my_deno[my_iq] = d2.sum()

    enum = np.empty((nQ, nmodes), dtype=my_enum.dtype)
    deno = np.empty(nQ)

    comm.Allgatherv(my_enum, (enum, sizes * nmodes))
    comm.Allgatherv(my_deno, (deno, sizes))

    return enum, deno

def grand_potential(e, kT=0.025, occupations='fd'):
    r"""Calculate (zeroth order of) grand potential.

    Parameters
    ----------
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    kT : float
        Smearing temperature.
    occupations : function
        Particle distribution as a function of energy divided by `kT`.

    Returns
    -------
    real
        Grand potential.
    """
    occupations = elphmod.occupations.smearing(occupations)

    x = e / kT

    prefactor = 2 * kT / np.prod(e.shape[:-1])

    if occupations is elphmod.occupations.fermi_dirac: # faster alternative
        return prefactor * np.log(occupations(-x)).sum()

    return prefactor * ((occupations(x) * x).sum() # U - mu N
        - occupations.entropy(x).sum()) # - T S

def first_order(e, g, kT=0.025, U=None, eps=1e-10, occupations='fd'):
    r"""Calculate first-order diagram of grand potential.

    Parameters
    ----------
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    g : ndarray
        Electron-phonon coupling for selected q point.
    kT : float
        Smearing temperature.
    U : ndarray
        Eigenvectors of Wannier Hamiltonian belonging to considered bands. If
        present (absent), the coupling `g` is assumed to be given in the same
        basis as `U` (eigenbasis of the Hamiltonian).
    eps : float
        Smallest allowed absolute value of divisor.
    occupations : function
        Particle distribution as a function of energy divided by `kT`.

    Returns
    -------
    complex
        Value of first-order diagram.
    """
    occupations = elphmod.occupations.smearing(occupations)

    nk = np.ones(3, dtype=int)
    nk[:len(e.shape[:-1])] = e.shape[:-1]

    nbnd = e.shape[-1]
    e = np.reshape(e, (*nk, nbnd))

    if U is not None:
        U = np.reshape(U, (*nk, -1, nbnd))
        norb = U.shape[3]

        g = np.reshape(g, (-1, *nk, norb, norb))
    else:
        g = np.reshape(g, (-1, *nk, nbnd, nbnd))

    f = occupations(e / kT)

    if U is not None:
        f = np.einsum('ijkam,ijkm,ijkbm->ijkab', U, f, U.conj())

        indices = 'ijkab,xijkba->x'
    else:
        indices = 'ijkm,xijkmm->x'

    return 2 / nk.prod() * np.einsum(indices, f, g)

def triangle(q, Q, e, gq, gQ, gqQ, kT=0.025, eps=1e-10, occupations='fd',
        fluctuations=False):
    r"""Calculate triangle diagram (third order of grand potential).

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
    q, Q : tuple
        q points in crystal coordinates :math:`q_i, q'_i \in [0, 2 \pi)`.
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    gq, gQ, gqQ : ndarray
        Electron-phonon coupling for given q points and their difference.
    kT : float
        Smearing temperature.
    eps : float
        Smallest allowed absolute value of divisor.
    occupations : function
        Particle distribution as a function of energy divided by `kT`.
    fluctuations : bool
        Return integrand too (for fluctuation analysis)?

    Returns
    -------
    complex
        Value of triangle.
    """
    occupations = elphmod.occupations.smearing(occupations)

    tmp = np.zeros(3)
    tmp[:len(q)] = q
    q = tmp

    tmp = np.zeros(3)
    tmp[:len(Q)] = Q
    Q = tmp

    nk_orig = e.shape[:-1]
    nk = np.ones(3, dtype=int)
    nk[:len(nk_orig)] = nk_orig

    nbnd = e.shape[-1]
    e = np.reshape(e, (*nk, nbnd))
    gq = np.reshape(gq, (*nk, nbnd, nbnd))
    gQ = np.reshape(gQ, (*nk, nbnd, nbnd))
    gqQ = np.reshape(gqQ, (*nk, nbnd, nbnd))

    x = e / kT

    f = occupations(x)
    d = occupations.delta(x) / kT
    p = occupations.delta_prime(x) / kT ** 2

    e = np.tile(e, (2, 2, 2, 1))
    f = np.tile(f, (2, 2, 2, 1))
    d = np.tile(d, (2, 2, 2, 1))

    scale = nk / (2 * np.pi)
    prefactor = 4 / nk.prod()

    chi = np.empty((nbnd, nbnd, nbnd, *nk), dtype=complex)

    q = np.round(q * scale).astype(int) % nk
    Q = np.round(Q * scale).astype(int) % nk

    k1 = slice(0, nk[0])
    k2 = slice(0, nk[1])
    k3 = slice(0, nk[2])

    kq1 = slice(q[0], q[0] + nk[0])
    kq2 = slice(q[1], q[1] + nk[1])
    kq3 = slice(q[2], q[2] + nk[2])

    kQ1 = slice(Q[0], Q[0] + nk[0])
    kQ2 = slice(Q[1], Q[1] + nk[1])
    kQ3 = slice(Q[2], Q[2] + nk[2])

    for a in range(nbnd):
        ea = e[k1, k2, k3, a]
        fa = f[k1, k2, k3, a]
        da = d[k1, k2, k3, a]

        for b in range(nbnd):
            eb = e[kq1, kq2, kq3, b]
            fb = f[kq1, kq2, kq3, b]
            db = d[kq1, kq2, kq3, b]

            for c in range(nbnd):
                ec = e[kQ1, kQ2, kQ3, c]
                fc = f[kQ1, kQ2, kQ3, c]
                dc = d[kQ1, kQ2, kQ3, c]

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
                chi[a, b, c][l] = -0.5 * p[..., a][l]

    for i in range(3):
        gqQ = np.roll(gqQ, shift=-Q[i], axis=i)

    chi_k = np.einsum('abcijk,ijkba,ijkca,ijkbc->ijkabc',
        chi, gq.conj(), gQ, gqQ)

    chi = prefactor * chi_k.sum()

    if fluctuations:
        return chi, 4 * chi_k.reshape((*nk_orig, nbnd, nbnd, nbnd))
    else:
        return chi

def fan_migdal_self_energy(k, e, w, g2, omega, kT=0.025, occupations='fd',
        comm=comm):
    r"""Calculate Fan-Migdal electron self-energy (to be tested).

    See Eq. (4) by Abramovitch et al., Phys. Rev. Mater. 7, 093801 (2023).

    Parameters
    ----------
    k : list of tuple
        List of k points in crystal coordinates :math:`k_i \in [0, 2 \pi)`.
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    w : ndarray
        Phonon dispersion on same uniform mesh as `e`.
    g2 : ndarray
        Squared electron-phonon coupling.
    omega : ndarray
        Frequency argument including small imaginary regulator.
    kT : float
        Smearing temperature.
    occupations : function
        Particle distribution as a function of energy divided by `kT`.

    Returns
    -------
    ndarray
        Fan-Migdal electron self-energy.
    """
    occupations = elphmod.occupations.smearing(occupations)

    nK = len(k)

    k_orig = k
    k = np.zeros((nK, 3))
    k[:, :len(k_orig[0])] = k_orig

    nq_orig = w.shape[:-1]
    nq = np.ones(3, dtype=int)
    nq[:len(nq_orig)] = nq_orig

    nbnd = e.shape[-1]
    nmodes = w.shape[-1]

    e = np.reshape(e, (*nq, 1, nbnd, 1, 1))
    w = np.reshape(w, (*nq, nmodes, 1, 1, 1))
    g2 = np.reshape(g2, (*nq, nmodes, nK, nbnd, nbnd, 1))
    omega = np.reshape(omega, (1, 1, 1, 1, 1, 1, -1))

    f = occupations(e / kT)
    N = elphmod.occupations.bose_einstein(w / kT)

    e = np.tile(e, (2, 2, 2, 1, 1, 1, 1))
    f = np.tile(f, (2, 2, 2, 1, 1, 1, 1))

    scale = nq / (2 * np.pi)
    prefactor = 1 / nq.prod()

    sizes, bounds = elphmod.MPI.distribute(nK, bounds=True, comm=comm)

    my_Sigma = np.empty((sizes[comm.rank], nbnd, omega.size), dtype=complex)

    status = elphmod.misc.StatusBar(sizes[comm.rank],
        title='calculate Fan-Migdal electron self-energy')

    for my_ik, ik in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        k1, k2, k3 = np.round(k[ik] * scale).astype(int) % nq

        kq1 = slice(k1, k1 + nq[0])
        kq2 = slice(k2, k2 + nq[1])
        kq3 = slice(k3, k3 + nq[2])

        fkq = f[kq1, kq2, kq3]
        ekq = e[kq1, kq2, kq3]

        domega = omega - ekq

        my_Sigma[my_ik] = prefactor * np.sum(g2[:, :, :, :, ik, :, :, :]
            * ((fkq + N) / (domega + w) + (1 - fkq + N) / (domega - w)),
            axis=(0, 1, 2, 3, 4))

        status.update()

    Sigma = np.empty((nK, nbnd, omega.size), dtype=complex)

    comm.Allgatherv(my_Sigma, (Sigma, comm.allgather(my_Sigma.size)))

    return Sigma

def green_kubo_conductivity(v, A, omega, kT=0.025, eps=1e-10, occupations='fd',
        dc_only=False, comm=comm):
    r"""Calculate Green-Kubo optical conductivity (to be tested).

    See Eq. (8) by Abramovitch et al., Phys. Rev. Mater. 7, 093801 (2023).
    Note that we have omitted the division by the unit-cell volume.

    Parameters
    ----------
    v : ndarray
        Fermi velocity on uniform mesh.
    A : ndarray
        Electronic spectral function.
    omega : ndarray
        Frequency argument excluding small imaginary regulator.
    kT : float
        Smearing temperature.
    eps : float
        Negligible difference between two floating-point numbers.
    occupations : function
        Particle distribution as a function of energy divided by `kT`.
    dc_only : bool
        Only compute DC resistivity tensor (zero-frequency limit)?

    Returns
    -------
    ndarray
        Green-Kubo optical conductivity.
    """
    occupations = elphmod.occupations.smearing(occupations)

    nbnd = A.shape[-2]
    nq = A.size // nbnd // len(omega)
    ndim = v.size // nq // nbnd

    vA = v.reshape((-1, 1, ndim)) * A.reshape((-1, len(omega), 1))

    x = omega[:, np.newaxis, np.newaxis] / kT
    d = occupations.delta(x) / kT

    prefactor = 4 * np.pi / nq # including e^2 = 2 and 2 / nq

    if dc_only:
        domega = elphmod.misc.differential(omega)[:, np.newaxis, np.newaxis]

        sigma = prefactor * np.sum(domega * d * np.sum(vA[:, :, :, np.newaxis]
            * vA[:, :, np.newaxis, :], axis=0), axis=0)
    else:
        domega = omega[1] - omega[0]

        if np.any(abs(np.diff(omega) - domega) > eps):
            info('Frequency sampling must be equidistant!', error=True)

        iw0 = np.argmin(abs(omega))

        if abs(omega[iw0]) > eps:
            info('Frequency sampling should include zero!', error=True)

        f = occupations(x)

        sizes, bounds = elphmod.MPI.distribute(len(omega), bounds=True,
            comm=comm)

        my_sigma = np.empty((sizes[comm.rank], ndim, ndim))

        status = elphmod.misc.StatusBar(sizes[comm.rank],
            title='calculate Green-Kubo optical conductivity')

        for my_iw, iw in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
            diw = iw - iw0
            diwm = max(0, -diw)
            diwp = max(0, diw)
            slmp = slice(diwm, len(omega) - diwp)
            slpm = slice(diwp, len(omega) - diwm)

            a = d if iw == iw0 else (f[slmp] - f[slpm]) / omega[iw]
            b = np.sum(vA[:, slmp, :, np.newaxis] * vA[:, slpm, np.newaxis, :],
                axis=0)

            my_sigma[my_iw] = prefactor * domega * np.sum(a * b, axis=0)

            status.update()

        sigma = np.empty((len(omega), ndim, ndim))

        comm.Allgatherv(my_sigma, (sigma, comm.allgather(my_sigma.size)))

    return sigma
