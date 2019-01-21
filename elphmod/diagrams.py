#/usr/bin/env python

import numpy as np

from . import MPI, occupations
comm = MPI.comm
info = MPI.info

kB = 8.61733e-5 # Boltzmann constant (eV/K)

def susceptibility(e, T=1.0, eta=1e-10, occupations=occupations.fermi_dirac):
    """Calculate real part of static electronic susceptibility.

        chi(q) = 2/N sum[k] [f(k+q) - f(k)] / [e(k+q) - e(k) + i eta]

    The resolution in q is limited by the resolution in k.

    Parameters
    ----------
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    T : float
        Smearing temperature in K.
    eta : float
        Absolute value of "infinitesimal" imaginary number in denominator.
    occupations : function
        Particle distribution as a function of energy divided by kT.

    Returns
    -------
    function
        Static electronic susceptibility as a function of q1, q2 in [0, 2pi).
    """
    nk, nk = e.shape

    kT = kB * T
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

    calculate_susceptibility.size = 1

    return calculate_susceptibility

def susceptibility2(e, T=1.0, nmats=1000, hyb_width=1.0, hyb_height=0.0):
    """Calculate the Lindhardt bubble using the Green's functions explicitly.

        chi = beta/4 - 1/beta sum[GG - 1/(i nu)^2]

    Only omega = 0 (static) calculation is performed.

    For the treatment of the 1/inu tail, see:

        Appendix B of the thesis of Hartmut Hafermann.

    Multiply by 2 for spin.

    The resolution in q is limited by the resolution in k.

    Original implementation by Erik G.C.P. van Loon.

    Parameters
    ----------
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    T : float
        Smearing temperature in K.
    nmats : int
        Number of fermionic Matsubara frequencies.
    hyb_width : float
        Width of box-shaped hybridization function.
    hyb_height : float
        Height of box-shaped hybridization function.

    Returns
    -------
    function
        Static electronic susceptibility as a function of q1, q2 in [0, 2pi).
    """
    nk, nk = e.shape

    kT = kB * T

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

    calculate_susceptibility.size = 1

    return calculate_susceptibility

def polarization(e, c, T=1.0, eps=1e-15, subspace=None,
        occupations=occupations.fermi_dirac):
    """Calculate RPA polarization in orbital basis (density-density).

        Pi(q, a, b) = 2/N sum[k, n, m]
            <k+q m|k+q a> <k a|k n> <k n|k b> <k+q b|k+q m>
            [f(k+q, m) - f(k, n)] / [e(k+q, m) - e(k, n)]

    The resolution in q is limited by the resolution in k.

    If 'subspace' is given, a cRPA calculation is performed. 'subspace' must be
    a boolean array with the same shape as 'e', where 'True' marks states of the
    target subspace, interactions between which are excluded.

    Parameters
    ----------
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    c : ndarray
        Coefficients for transform to orbital basis. These are given by the
        eigenvectors of the Wannier Hamiltonian.
    T : float
        Smearing temperature in K.
    eps : float
        Smallest allowed absolute value of divisor.
    subspace : ndarray or None
        Boolean array to select k points and/or bands in cRPA target subspace.
    occupations : function
        Particle distribution as a function of energy divided by kT.

    Returns
    -------
    function
        RPA polarization in orbital basis as a function of q1, q2 in [0, 2pi).
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

    kT = kB * T
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

    calculate_polarization.size = nb

    return calculate_polarization

def phonon_self_energy(q, e, g2, T=100.0, eps=1e-15,
        occupations=occupations.fermi_dirac, status=True):
    """Calculate phonon self-energy.

        Pi(q, nu) = 2/N sum[k] |g(q, nu, k)|^2
            [f(k+q) - f(k)] / [e(k+q) - e(k)]

    Parameters
    ----------
    q : list of 2-tuples
        Considered q points defined via crystal coordinates q1, q2 in [0, 2pi).
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    g2 : ndarray
        Squared electron-phonon coupling.
    T : float
        Smearing temperature in K.
    eps : float
        Smallest allowed absolute value of divisor.
    occupations : function
        Particle distribution as a function of energy divided by kT.
    status : bool
        Print status messages during the calculation?

    Returns
    -------
    ndarray
        Phonon self-energy.
    """
    nk, nk = e.shape
    nQ, nb, nk, nk = g2.shape

    kT = kB * T
    x = e / kT

    f = occupations(x)
    d = occupations.delta(x) / (-kT)

    e = np.tile(e, (2, 2))
    f = np.tile(f, (2, 2))

    scale = nk / (2 * np.pi)
    prefactor = 2.0 / nk ** 2

    sizes, bounds = MPI.distribute(nQ, bounds=True)

    my_Pi = np.empty((sizes[comm.rank], nb), dtype=complex)

    if status:
        info('Pi(%3s, %3s, %3s) = ...' % ('q1', 'q2', 'nu'))

    chi = np.empty((nk, nk))

    for my_iq, iq in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        q1 = int(round(q[iq, 0] * scale)) % nk
        q2 = int(round(q[iq, 1] * scale)) % nk

        df = f[q1:q1 + nk, q2:q2 + nk] - f[:nk, :nk]
        de = e[q1:q1 + nk, q2:q2 + nk] - e[:nk, :nk]

        ok = abs(de) > eps

        chi[ ok] = df[ok] / de[ok]
        chi[~ok] = d[~ok]

        for nu in range(nb):
            my_Pi[my_iq, nu] = prefactor * np.sum(g2[iq, nu] * chi)

            if status:
                print('Pi(%3d, %3d, %3d) = %9.2e%+9.2ei'
                    % (q1, q2, nu, my_Pi[my_iq, nu].real, my_Pi[my_iq, nu].imag))

    Pi = np.empty((nQ, nb), dtype=complex)

    comm.Allgatherv(my_Pi, (Pi, sizes * nb))

    return Pi

def phonon_self_energy2(q, e, g2, T=100.0, nmats=1000, hyb_width=1.0,
        hyb_height=0.0, status=True, GB=4.0):
    """Calculate phonon self-energy using the Green's functions explicitly.

    Parameters
    ----------
    q : list of 2-tuples
        Considered q points defined via crystal coordinates q1, q2 in [0, 2pi).
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    g2 : ndarray
        Squared electron-phonon coupling.
    T : float
        Smearing temperature in K.
    nmats : int
        Number of fermionic Matsubara frequencies.
    hyb_width : float
        Width of box-shaped hybridization function.
    hyb_height : float
        Height of box-shaped hybridization function.
    status : bool
        Print status messages during the calculation?
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

    kT = kB * T

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

    if status:
        info('Pi(%3s, %3s, %3s) = ...' % ('q1', 'q2', 'nu'))

    for my_iq, iq in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        q1 = int(round(q[iq, 0] * scale)) % nk
        q2 = int(round(q[iq, 1] * scale)) % nk

        Gk  = G[:, :nk, :nk]
        Gkq = G[:, q1:q1 + nk, q2:q2 + nk]

        chi = prefactor * np.sum(Gk * Gkq, axis=0).real + tail

        for nu in range(nb):
            my_Pi[my_iq, nu] = np.sum(g2[iq, nu] * chi)

            if status:
                print('Pi(%3d, %3d, %3d) = %9.2e%+9.2ei'
                    % (q1, q2, nu, my_Pi[my_iq, nu].real, my_Pi[my_iq, nu].imag))

    Pi = np.empty((nQ, nb), dtype=complex)

    comm.Allgatherv(my_Pi, (Pi, sizes * nb))

    return Pi

def renormalize_coupling(q, e, g, W, U, T=100.0, eps=1e-15,
        occupations=occupations.fermi_dirac, pre=2, dd=False, einsum=True,
        status=True):
    """Calculate renormalized electron-phonon coupling.

    g'(k, q, i, x) = g(k, q, i, x) + 2/N sum[k'] g(k', q, i, x)
        [f(k'+q) - f(k')] / [e(k'+q) - e(k')] W(k, k', q)

    Parameters
    ----------
    q : list of 2-tuples
        Considered q points defined via crystal coordinates q1, q2 in [0, 2pi).
    e : ndarray
        Electron dispersion on uniform mesh. The Fermi level must be at zero.
    g : ndarray
        Bare electron-phonon coupling.
    W : ndarray
        Dressed Coulomb interaction.
    U : ndarray
        Eigenvectors of Wannier Hamiltonian belonging to considered band.
    T : float
        Smearing temperature in K.
    eps : float
        Smallest allowed absolute value of divisor.
    occupations : function
        Particle distribution as a function of energy divided by kT.
    pre : int
        Spin prefactor 1 or 2? Used for debugging only.
    dd : bool
        Consider only density-density terms of Coulomb interaction. The shape
        of the parameter W depends on this parameter.
    einsum : bool
        Use numpy.einsum for k' summations in expression for Feynman diagram?
        (It seems that its implementations range from very fast to very slow.)
    status : bool
        Print status messages during the calculation?

    Returns
    -------
    ndarray
        Dressed electron-phonon coupling.
    """
    nk, nk = e.shape
    nQ, nmodes, nk, nk = g.shape

    if dd:
        nq, nq, nbnd, nbnd = W.shape
    else:
        nq, nq, nbnd, nbnd, nbnd, nbnd = W.shape

    nk, nk, nbnd = U.shape

    kT = kB * T
    x = e / kT

    f = occupations(x)
    d = occupations.delta(x) / (-kT)

    e = np.tile(e, (2, 2))
    f = np.tile(f, (2, 2))
    U = np.tile(U, (2, 2, 1))

    scale_k = nk / (2 * np.pi)
    scale_q = nq / (2 * np.pi)
    prefactor = pre / nk ** 2

    sizes, bounds = MPI.distribute(nQ, bounds=True)

    my_g_ = np.empty((sizes[comm.rank], nmodes, nk, nk), dtype=complex)

    dfde = np.empty((nk, nk))

    for my_iq, iq in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        if status:
            print('Renormalize coupling for q point %d..' % (iq + 1))

        q1 = int(round(q[iq, 0] * scale_q)) % nq
        q2 = int(round(q[iq, 1] * scale_q)) % nq
        Q1 = int(round(q[iq, 0] * scale_k)) % nk
        Q2 = int(round(q[iq, 1] * scale_k)) % nk

        k1 = slice(0, nk)
        k2 = slice(0, nk)

        kq1 = slice(Q1, Q1 + nk)
        kq2 = slice(Q2, Q2 + nk)

        df = f[kq1, kq2] - f[k1, k2]
        de = e[kq1, kq2] - e[k1, k2]

        ok = abs(de) > eps

        dfde[ ok] = df[ok] / de[ok]
        dfde[~ok] = d[~ok]

        if einsum:
            if dd:
                indices = 'aij,ik,kl,ijk,ijk,ijk,bcl,bcl->abc'

                # g[iq, i, k1', k2'] * dfde[k1', k2'] * W[q1, q2, a, b]
                #       a  i    j           i    j                k  l
                # * U[kq1', kq2', a] * U[k1', k2', a].conj()
                #     i     j     k      i    j    k
                # * U[k1,   k2,   b] * U[kq1, kq2, b].conj()
                #     b     c     l      b    c    l
            else:
                indices = 'aij,ij,klmn,ijk,ijl,bcm,bcn->abc'

                # g[iq, i, k1', k2'] * dfde[k1', k2'] * W[q1, q2, a, b, c, d]
                #       a  i    j           i    j                k  l  m  n
                # * U[kq1', kq2', a] * U[k1', k2', b].conj()
                #     i     j     k      i    j    l
                # * U[k1,   k2,   c] * U[kq1, kq2, d].conj()
                #     b     c     m      b    c    n

            my_g_[iq] = g[iq] + prefactor * np.einsum(indices,
                g[iq], dfde, W[q1, q2],
                U[kq1, kq2], U[k1, k2].conj(),
                U[k1, k2], U[kq1, kq2].conj())

        else:
            tmp = np.empty((nbnd, nbnd, nmodes), dtype=complex)

            for a in range(nbnd):
                for b in range(nbnd):
                    if dd and a != b:
                        continue

                    dfdeab = dfde * U[kq1, kq2, a] * U[k1, k2, b].conj()

                    for i in range(nmodes):
                        tmp[a, b, i] = (g[iq, i] * dfdeab).sum()

            if dd:
                indices = 'kka,kl,bcl,bcl->abc'

                # tmp[a, b, i] * W[q1, q2, a, b]
                #     k  k  a              k  l
                # * U[k1, k2, c] * U[kq1, kq2, d].conj()
                #     b   c   l      b    c    l

            else:
                indices = 'kla,klmn,bcm,bcn->abc'

                # tmp[a, b, i] * W[q1, q2, a, b, c, d]
                #     k  l  a              k  l  m  n
                # * U[k1, k2, c] * U[kq1, kq2, d].conj()
                #     b   c   m      b    c    n

            my_g_[iq]  = g[iq] + prefactor * np.einsum(indices,
                tmp, W[q1, q2], U[k1, k2], U[kq1, kq2].conj())

    g_ = np.empty((nQ, nmodes, nk, nk), dtype=complex)

    comm.Allgatherv(my_g_, (g_, sizes * nmodes * nk * nk))

    return g_
