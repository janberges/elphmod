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

    e = np.tile(e, (2, 2))
    f = np.tile(f, (2, 2))

    def calculate_susceptibility(q1=0, q2=0):
        q1 = int(round(q1 / (2 * np.pi) * nk)) % nk
        q2 = int(round(q2 / (2 * np.pi) * nk)) % nk

        de = e[q1:q1 + nk, q2:q2 + nk] - e[:nk, :nk]
        df = f[q1:q1 + nk, q2:q2 + nk] - f[:nk, :nk]

        return 2 * np.sum(df * de / (de * de + eta * eta)) / nk ** 2

    return calculate_susceptibility
