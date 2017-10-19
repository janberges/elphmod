#/usr/bin/env python

import numpy as np

def hexDOS(energies):
    """Calculate DOS from energies on triangular mesh (2D tetrahedron).

    Integration over all energies yields unity."""

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

        return D / N ** 2

    return np.vectorize(DOS)

def hexa2F(energies, couplings):
    """Calculate a2F from energies and coupling.

    Integration over all energies yields the arithmetic mean of the coupling."""

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

        return D / N ** 2

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
