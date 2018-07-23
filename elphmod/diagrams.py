#/usr/bin/env python

import numpy as np

from . import MPI
comm = MPI.comm

def susceptibility(e, T=1.0, eta=1e-10):
    """Calculate real part of static electronic susceptibility

        chi(q) = 2/N sum[k] [f(k+q) - f(k)] / [e(k+q) - e(k) + i eta].

    The resolution in q is limited by the resolution in k."""

    nk, nk = e.shape

    T *= 8.61733e-5 # K to eV

    f = 1 / (np.exp(e / T) + 1)
    d = 1 / (2 * T * (np.cosh(e / T) + 1))

    DOS = d.sum()

    e = np.tile(e, (2, 2))
    f = np.tile(f, (2, 2))

    scale = nk / (2 * np.pi)
    eta2 = eta ** 2
    prefactor = 2.0 / nk ** 2

    def calculate_susceptibility(q1=0, q2=0):
        q1 = int(round(q1 * scale)) % nk
        q2 = int(round(q2 * scale)) % nk

        if q1 == q2 == 0:
            return -prefactor * DOS

        df = f[q1:q1 + nk, q2:q2 + nk] - f[:nk, :nk]
        de = e[q1:q1 + nk, q2:q2 + nk] - e[:nk, :nk]

        return prefactor * np.sum(df * de / (de * de + eta2))

    calculate_susceptibility.size = 1

    return calculate_susceptibility

def phonon_self_energy(e, g2, T=1.0, eta=1e-10):
    """Calculate real part of the phonon self-energy

        Pi(q, nu) = 2/N sum[k] |g(q, nu, k)|^2
            [f(k+q) - f(k)] / [e(k+q) - e(k) + i eta].

    The resolution in q is limited by the resolution in k and q of the input."""

    nk, nk = e.shape
    nq, nq, nk, nk = g2.shape

    T *= 8.61733e-5 # K to eV

    f = 1 / (np.exp(e / T) + 1)

    e = np.tile(e, (2, 2))
    f = np.tile(f, (2, 2))

    scale_k = nk / (2 * np.pi)
    scale_q = nq / (2 * np.pi)

    eta2 = eta ** 2
    prefactor = 2.0 / nk ** 2

    def calculate_phonon_self_energy(q1=0, q2=0):
        Q1 = int(round(q1 * scale_q)) % nq
        Q2 = int(round(q2 * scale_q)) % nq

        q1 = int(round(q1 * scale_k)) % nk
        q2 = int(round(q2 * scale_k)) % nk

        df = f[q1:q1 + nk, q2:q2 + nk] - f[:nk, :nk]
        de = e[q1:q1 + nk, q2:q2 + nk] - e[:nk, :nk]

        return prefactor * np.sum(g2[Q1, Q2] * df * de / (de * de + eta2))

    calculate_phonon_self_energy.size = 1

    return calculate_phonon_self_energy
