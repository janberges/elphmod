#/usr/bin/env python

import numpy as np

deg = np.pi / 180

def rotate(vector, angle):
    """Rotate vector."""

    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)],
        ]).dot(vector)

def reciprocals(t1, t2):
    """Get translation vectors of reciprocal lattice."""

    u1 = rotate(t2, -90 * deg)
    u2 = rotate(t1, +90 * deg)

    u1 /= t1.dot(u1)
    u2 /= t2.dot(u2)

    return u1, u2

# define real and reciprocal (without 2 pi) translation vectors:

#1 Quantum ESPRESSO:

t1 = np.array([1.0, 0.0])
t2 = rotate(t1, 120 * deg)

u1, u2 = reciprocals(t1, t2)

#2 Brillouin-zone plots:

T1 = rotate(t1, -30 * deg)
T2 = rotate(t2, -30 * deg)

U1, U2 = reciprocals(T1, T2)

def images(k1, k2, nk):
    """Get equivalents k points."""

    points = set()

    k = k1 * u1 + k2 * u2

    for reflect in False, True:
        for angle in range(0, 360, 60):
            K = rotate(k, angle * deg)

            if reflect:
                K[0] *= -1

            K1 = int(round(np.dot(K, t1))) % nk
            K2 = int(round(np.dot(K, t2))) % nk

            points.add((K1, K2))

    return points

def interpolate(mesh, q1, q2, angle=30):
    """Perform linear interpolation on triangular lattice."""

    nq1, nq2 = mesh.shape

    q01 = int(q1 % nq1)
    q02 = int(q2 % nq2)
    dq1 = q1 % 1
    dq2 = q2 % 1

    if angle == 30:
        A = mesh[(q01 + 1) % nq1, q02]
        B = mesh[q01, (q02 + 1) % nq2]

        if dq1 + dq2 > 1:
            C = mesh[(q01 + 1) % nq1, (q02 + 1) % nq2]
            return (1 - dq2) * A + (1 - dq1) * B + (dq1 + dq2 - 1) * C
        else:
            C = mesh[q01, q02]
            return dq1 * A + dq2 * B + (1 - dq1 - dq2) * C

    elif angle == 60:
        A = mesh[q01, q02]
        B = mesh[(q01 + 1) % nq1, (q02 + 1) % nq2]

        if dq1 > dq2:
            C = mesh[(q01 + 1) % nq1, q02]
            return (1 - dq1) * A + dq2 * B + (dq1 - dq2) * C
        else:
            C = mesh[q01, (q02 + 1) % nq2]
            return (1 - dq2) * A + dq1 * B + (dq2 - dq1) * C

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

    return np.array(path), x

