#/usr/bin/env python

import numpy as np

from . import MPI
comm = MPI.comm

def hexDOS(energies):
    """Calculate DOS from energies on triangular mesh (2D tetrahedron method).

    Integration over all energies yields unity.

    Derivation: The density of states can be expressed as

        DOS(E) = 1/V integral[W(k) = E] dk / |grad W(k)|,

    where V is the volume of a reciprocal unit cell, W(k) is the dispersion
    relation and the integral is over an iso-energy surface/line in k space.

    In the following, consider a two dimensional reciprocal unit cell which can
    be divided into 2 x N x N equilateral triangles of reciprocal side length a,
    on each of which the energy is interpolated linearly. On a triangle with the
    energies A, B, C at its corners, the gradient of the energy plane is

        |grad W| = 2/[sqrt(3) a] sqrt(A^2 + B^2 + C^2 - AB - AC - BC).

    For the special case A < B < E < C, the reciprocal length of the E isoline
    within the triangle reads

        dk = a sqrt(A^2 + B^2 + C^2 - AB - AC - BC) (C - E) / [(C - A) (C - B)].

    Taking into account that V = N^2 a^2 sqrt(3)/2, one finds the contribution
    of this triangle to the density of states:

        1/N^2 (C - E) / [(C - A) (C - B)]."""

    N, N = energies.shape

    triangles = [
        sorted([
            energies[(i + k) % N, (j + k) % N],
            energies[(i + 1) % N,  j,        ],
            energies[ i,          (j + 1) % N],
            ])
        for i in range(N)
        for j in range(N)
        for k in range(2)
        if comm.rank == ((i * N + j) * 2 + k) % comm.size
        ]

    D = np.empty(len(triangles))

    def DOS(E):
        for n, (A, B, C) in enumerate(triangles):
            if A < E <= B:
                if E == B == C:
                    D[n] = 0.5 / (E - A)
                else:
                    D[n] = (E - A) / (B - A) / (C - A)

            elif B <= E < C:
                if E == A == B:
                    D[n] = 0.5 / (C - E)
                else:
                    D[n] = (C - E) / (C - A) / (C - B)

            elif E == A == B == C:
                D[n] = float('inf')

            else:
                D[n] = 0.0

        return comm.allreduce(D.sum()) / N ** 2

    return np.vectorize(DOS)

def hexa2F(energies, couplings):
    """Calculate a2F from energies and coupling.

    Integration over all energies yields the arithmetic mean of the coupling.

    Note that it may be more convenient to calculate the mass renormalization

        lambda[n] = integral[w > 0] dw 2w a^2F(w) / (w^2 + w[n]^2)
                  = N(0) sum[q, nu] 2w[q, nu] g^2[q nu] / (w[q, nu]^2 + w[n]^2)

    directly from energies and couplings, without integrating this function."""

    N, N = energies.shape

    triangles = [
        tuple(zip(*sorted([
            ((i + k) % N, (j + k) % N),
            ((i + 1) % N,  j,        ),
            ( i,          (j + 1) % N),
            ], key=lambda x: energies[x])))
        for i in range(N)
        for j in range(N)
        for k in range(2)
        if comm.rank == ((i * N + j) * 2 + k) % comm.size
        ]

    triangles = [(energies[v], couplings[v]) for v in triangles]

    D = np.empty(len(triangles))

    def a2F(E):
        for n, ((A, B, C), (a, b, c)) in enumerate(triangles):
            if A < E <= B:
                if E == B == C:
                    D[n] = 0.5 / (E - A) * 0.5 * (b + c)
                else:
                    D[n] = (E - A) / (B - A) / (C - A) * 0.5 * (
                        ((E - A) * b + (B - E) * a) / (B - A) +
                        ((E - A) * c + (C - E) * a) / (C - A))

            elif B <= E < C:
                if E == A == B:
                    D[n] = 0.5 / (C - E) * 0.5 * (a + b)
                else:
                    D[n] = (C - E) / (C - A) / (C - B) * 0.5 * (
                        ((C - E) * a + (E - A) * c) / (C - A) +
                        ((C - E) * b + (E - B) * c) / (C - B))

            elif E == A == B == C:
                D[n] = float('inf')

            else:
                D[n] = 0.0

        return comm.allreduce(D.sum()) / N ** 2

    return np.vectorize(a2F)

def double_delta(x, y):
    """Calculate points where x = y = z."""

    N, N = x.shape

    triangles = [
        np.array([
            (i + k, j + k),
            (i + 1, j    ),
            (i,     j + 1),
        ])
        for i in range(N)
        for j in range(N)
        for k in range(2)
        if comm.rank == ((i * N + j) * 2 + k) % comm.size
        ]

    indices = [tuple(zip(*v % N)) for v in triangles]

    triangles = [(x[i], y[i], v) for i, v in zip(indices, triangles)]

    D = []

    def dd(z):
        for (A, B, C), (a, b, c), (X, Y, Z) in triangles:
            denum = a * B - A * b - a * C + A * c + b * C - B * c

            if denum == 0:
                continue

            n1 = (A * c - a * C + (a - A + C - c) * z) / denum
            n2 = (a * B - A * b + (A - a + b - B) * z) / denum
            n3 = n1 + n2

            if X[0] == Y[0] and 0 <= n1 <= 1 and 0 <= n2 <= 1 and 0 <= n3 <= 1\
            or X[0] == Z[0] and 0 <  n1 <  1 and 0 <  n2 <  1 and 0 <  n3 <  1:
                D.append(X + n1 * (Y - X) + n2 * (Z - X))

        return np.reshape(comm.gather(D), (-1, 2))

    return dd

def simpleDOS(energies, smearing):
    """Calculate DOS from representative energy sample (Lorentzian sum).

    Integration over all energies yields unity."""

    const = smearing / np.pi / energies.size

    def DOS(energy):
        return np.sum(const / (smearing ** 2 + (energy - energies) ** 2))

    return np.vectorize(DOS)

if __name__ == '__main__':
    # Test DOS functions for tight-binding band of graphene:

    import matplotlib.pyplot as plt

    k = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    E = np.empty(k.shape * 2)

    for i, p in enumerate(k):
        for j, q in enumerate(k):
            E[i, j] = -np.sqrt(3 + 2 * (np.cos(p) + np.cos(q) + np.cos(p + q)))

    e = np.linspace(E.min(), E.max(), 300)

    plt.fill_between(e, 0, hexDOS(E)(e), facecolor='lightgray')
    plt.plot(e, simpleDOS(E, smearing=0.02)(e), color='red')
    plt.show()
