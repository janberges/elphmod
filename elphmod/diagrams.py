#/usr/bin/env python

import numpy as np

from . import MPI, occupations
comm = MPI.comm
info = MPI.info

kB = 8.61733e-5 # Boltzmann constant (eV/K)

def susceptibility(e, T=1.0, eta=1e-10):
    """Calculate real part of static electronic susceptibility

        chi(q) = 2/N sum[k] [f(k+q) - f(k)] / [e(k+q) - e(k) + i eta].

    The resolution in q is limited by the resolution in k."""

    nk, nk = e.shape

    kT = kB * T
    x = e / kT

    f = occupations.fermi_dirac(x)
    d = occupations.fermi_dirac_delta(x).sum() / kT

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

def polarization(e, c, T=1.0, i0=1e-10j):
    """Calculate RPA polarization

        Pi(q, a, b) = 2/N sum[k] <k+q|a><a|k><k|b><b|k+q>
            [f(k+q) - f(k)] / [e(k+q) - e(k) + i0].

    The resolution in q is limited by the resolution in k."""

    nk, nk = e.shape
    nk, nk, nb = c.shape # c[k1, k2, a] = <a|k>

    kT = kB * T
    x = e / kT

    f = occupations.fermi_dirac(x)

    e = np.tile(e, (2, 2))
    f = np.tile(f, (2, 2))
    c = np.tile(c, (2, 2, 1))

    scale = nk / (2 * np.pi)
    prefactor = 2.0 / nk ** 2

    def calculate_polarization(q1=0, q2=0):
        q1 = int(round(q1 * scale)) % nk
        q2 = int(round(q2 * scale)) % nk

        df = f[q1:q1 + nk, q2:q2 + nk] - f[:nk, :nk]
        de = e[q1:q1 + nk, q2:q2 + nk] - e[:nk, :nk]

        cc = c[:nk, :nk] * c[q1:q1 + nk, q2:q2 + nk].conj()

        Pi = np.empty((nb, nb), dtype=complex)

        for a in range(nb):
            cca = cc[:, :, a]

            for b in range(nb):
                ccb = cc[:, :, b].conj()

                Pi[a, b] = np.sum(cca * ccb * df / (de + i0))

        return prefactor * Pi

    calculate_polarization.size = nb

    return calculate_polarization

def phonon_self_energy(q, e, g2, T=100.0, i0=1e-10j,
        occupations=occupations.fermi_dirac):
    """Calculate phonon self-energy

        Pi(q, nu) = 2/N sum[k] |g(q, nu, k)|^2
            [f(k+q) - f(k)] / [e(k+q) - e(k) + i0]."""

    nk, nk = e.shape
    nQ, nb, nk, nk = g2.shape

    f = occupations(e / (kB * T))

    e = np.tile(e, (2, 2))
    f = np.tile(f, (2, 2))

    scale = nk / (2 * np.pi)
    prefactor = 2.0 / nk ** 2

    sizes, bounds = MPI.distribute(nQ, bounds=True)

    my_Pi = np.empty((sizes[comm.rank], nb), dtype=complex)

    info('Pi(%3s, %3s, %3s) = ...' % ('q1', 'q2', 'nu'))

    for my_iq, iq in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        q1 = int(round(q[iq, 0] * scale)) % nk
        q2 = int(round(q[iq, 1] * scale)) % nk

        df = f[q1:q1 + nk, q2:q2 + nk] - f[:nk, :nk]
        de = e[q1:q1 + nk, q2:q2 + nk] - e[:nk, :nk]

        chi = df / (de + i0)

        for nu in range(nb):
            my_Pi[my_iq, nu] = prefactor * np.sum(g2[iq, nu] * chi)

            print('Pi(%3d, %3d, %3d) = %9.2e%+9.2ei'
                % (q1, q2, nu, my_Pi[my_iq, nu].real, my_Pi[my_iq, nu].imag))

    Pi = np.empty((nQ, nb), dtype=complex)

    comm.Allgatherv(my_Pi, (Pi, sizes * nb))

    return Pi
