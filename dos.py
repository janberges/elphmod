#/usr/bin/env python

import numpy as np

def hexDOS(energies, comm=None):
    """Calculate DOS from energies on triangular mesh (2D tetrahedron).

    Integration over all energies yields unity.

    Derivation: The density of states can be expressed as

        DOS(E) = 1/V integral[W(k) = E] dk / |grad W(k)|,

    where V is the volume of a reciprocal unit cell, W(k) is the dispersion
    relation and the integral is over an iso-energy surface/line in k space.

    In the following, consider a two dimensional reciprocal unit cell which can
    be divided into 2 x N x N equilateral triangles of reciprocal side length a,
    on each of which the energy is interpolated linearly. On a triangle with the
    energies A, B, C at its corners, the squared gradient of the energy plane is

        |grad W|^2 = 4/3a^2 (A^2 + B^2 + C^2 - AB - AC - BC).

    For the special case A < B < E < C, the square of the reciprocal length of
    the E isoline within the triangle reads

        dk^2 = a^2 (A^2 + B^2 + C^2 - AB - AC - BC) (E-A)^2 / [(A-C) (B-C)]^2.

    Taking into account that V = N^2 a^2 sqrt(3)/2, one finds the contribution
    of this triangle to the density of states:

        1/N^2 (E - C) / [(A - C) (B - C)."""

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
        if comm is None or comm.rank == ((i * N + j) * 2 + k) % comm.size
        ]

    def DOS(E):
        D = 0.0

        for A, B, C in triangles:
            if A < E <= B:
                if E == B == C:
                    D += 0.5 / (E - A)
                else:
                    D += (E - A) / (B - A) / (C - A)

            elif B <= E < C:
                if E == A == B:
                    D += 0.5 / (C - E)
                else:
                    D += (C - E) / (C - A) / (C - B)

            elif E == A == B == C:
                return float('inf')

        return (D if comm is None else comm.allreduce(D)) / N ** 2

    return np.vectorize(DOS)

def hexa2F(energies, couplings, comm=None):
    """Calculate a2F from energies and coupling.

    Integration over all energies yields the arithmetic mean of the coupling.

    Note that it may be more convenient to calculate the mass renormalization

        lambda[n] = integral[w > 0] dw 2w a^2F(w) / (w^2 + w[n]^2)
                  = N(0) sum[q, nu] 2w[q, nu] g^2[q nu] / (w[q, nu]^2 + w[n]^2)

    directly from energies and couplings, without integrating this function."""

    N, N = energies.shape

    triangles = [
        zip(*sorted([
            ((i + k) % N, (j + k) % N),
            ((i + 1) % N,  j,        ),
            ( i,          (j + 1) % N),
            ], key=lambda x: energies[x]))
        for i in range(N)
        for j in range(N)
        for k in range(2)
        if comm is None or comm.rank == ((i * N + j) * 2 + k) % comm.size
        ]

    triangles = [(energies[v], couplings[v]) for v in triangles]

    def a2F(E):
        D = 0.0

        for (A, B, C), (a, b, c) in triangles:
            if A < E <= B:
                if E == B == C:
                    D += 0.5 / (E - A) * 0.5 * (b + c)
                else:
                    D += (E - A) / (B - A) / (C - A) * 0.5 * (
                        ((E - A) * b + (B - E) * a) / (B - A) +
                        ((E - A) * c + (C - E) * a) / (C - A))

            elif B <= E < C:
                if E == A == B:
                    D += 0.5 / (C - E) * 0.5 * (a + b)
                else:
                    D += (C - E) / (C - A) / (C - B) * 0.5 * (
                        ((C - E) * a + (E - A) * c) / (C - A) +
                        ((C - E) * b + (E - B) * c) / (C - B))

            elif E == A == B == C:
                return float('inf')

        return (D if comm is None else comm.allreduce(D)) / N ** 2

    return np.vectorize(a2F)

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
