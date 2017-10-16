#/usr/bin/env python

import numpy as np

def GMKG(N=30):
    """Generate path Gamma-M-K-Gamma through Brillouin zone."""

    G = 2 * np.pi * np.array([0.0, 0.0])
    M = 2 * np.pi * np.array([1.0, 0.0]) / 2
    K = 2 * np.pi * np.array([1.0, 1.0]) / 3

    L1 = np.sqrt(3)
    L2 = 1.0
    L3 = 2.0

    N1 = int(round(N * L1))
    N2 = int(round(N * L2))
    N3 = int(round(N * L3)) + 1

    def line(k1, k2, N=100, endpoint=True):
        q1 = np.linspace(k1[0], k2[0], N, endpoint)
        q2 = np.linspace(k1[1], k2[1], N, endpoint)

        return zip(q1, q2)

    path = line(G, M, N1, False) \
         + line(M, K, N2, False) \
         + line(K, G, N3, True)

    x = np.empty(N1 + N2 + N3)

    x[      0:N1          ] = np.linspace(      0, L1,           N1, False)
    x[     N1:N1 + N2     ] = np.linspace(     L1, L1 + L2,      N2, False)
    x[N2 + N1:N1 + N2 + N3] = np.linspace(L2 + L1, L1 + L2 + L3, N3, True)

    return path, x

