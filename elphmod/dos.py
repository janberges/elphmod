#!/usr/bin/env python3

# Copyright (C) 2017-2022 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import numpy as np

from . import bravais, misc, MPI
comm = MPI.comm

def hexDOS(energies, minimum=None, maximum=None):
    r"""Calculate DOS from energies on triangular mesh (2D tetrahedron method).

    Parameters
    ----------
    energies : ndarray
        Energies on triangular mesh.
    minimum, maximum : float, optional
        Energy window of interest (for efficiency).

    Returns
    -------
    function
        Density of states as a function of energy.

    Notes
    -----
    Integration over all energies yields unity.

    Derivation: The density of states can be expressed as

    .. math::

        \rho(E) = \frac 1 V \int_{W(k) = E} \frac{\D k}{|\vec \nabla W(k)|},

    where :math:`V` is the volume of a reciprocal unit cell, :math:`W(k)` is
    the dispersion relation and the integral is over an energy isosurface/line
    in k space.

    In the following, consider a two dimensional reciprocal unit cell which can
    be divided into :math:`2 \times N \times N` equilateral triangles of
    reciprocal side length :math:`a`, on each of which the energy is
    interpolated linearly. On a triangle with the energies :math:`A, B, C` at
    its corners, the gradient of the energy plane is

    .. math::

        |\vec \nabla W| = \frac 2 {\sqrt 3 a}
            \sqrt{A^2 + B^2 + C^2 - A B - A C - B C}.

    For the special case :math:`A < B < E < C`, the reciprocal length of the
    :math:`E` isoline within the triangle reads

    .. math::

        \D k = a \sqrt{A^2 + B^2 + C^2 - A B - A C - B C}
            \frac{C - E}{(C - A) (C - B)}.

    Taking into account that :math:`V = N^2 a^2 \sqrt 3 / 2`, one finds the
    contribution of this triangle to the density of states:

    .. math::

        \frac 1 {N^2} \frac{C - E}{(C - A) (C - B)}.
    """
    N, N = energies.shape

    triangles = [
        sorted([
            energies[(i + k) % N, (j + k) % N],
            energies[(i + 1) % N, j],
            energies[i, (j + 1) % N],
            ])
        for i in range(N)
        for j in range(N)
        for k in range(2)
        ]

    if minimum is not None:
        triangles = [v for v in triangles if v[2] >= minimum]

    if maximum is not None:
        triangles = [v for v in triangles if v[0] <= maximum]

    triangles = [v for n, v in enumerate(triangles)
        if comm.rank == n % comm.size]

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

def hexa2F(energies, couplings, minimum=None, maximum=None):
    r"""Calculate :math:`\alpha^2 F` from energies and coupling.

    Parameters
    ----------
    energies : ndarray
        Energies on triangular mesh.
    couplings : ndarray
        Couplings on triangular mesh.
    minimum, maximum : float, optional
        Energy window of interest (for efficiency).

    Returns
    -------
    function
        :math:`\alpha^2 F` as a function of energy.

    Notes
    -----
    Integration over all energies yields the arithmetic mean of the coupling.

    Note that it may be more convenient to calculate the mass renormalization

    .. math::
        \lambda_n = \int_0^\infty \D \omega
            \frac{2 \omega \alpha^2 F(\omega)}
                {\omega^2 + \omega_n^2}
            = N(0) \sum_{\vec q \nu}
            \frac{2 \omega_{\vec q \nu} g^2_{\vec q \nu}}
                {\omega_{\vec q \nu}^2 + \omega_n^2}

    directly from energies and couplings, without integrating this function."""

    N, N = energies.shape

    triangles = [
        sorted([
            ((i + k) % N, (j + k) % N),
            ((i + 1) % N, j),
            (i, (j + 1) % N),
            ], key=lambda x: energies[x])
        for i in range(N)
        for j in range(N)
        for k in range(2)
        ]

    if minimum is not None:
        triangles = [v for v in triangles if energies[v[2]] >= minimum]

    if maximum is not None:
        triangles = [v for v in triangles if energies[v[0]] <= maximum]

    triangles = [tuple(zip(*v)) for n, v in enumerate(triangles)
        if comm.rank == n % comm.size]

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

def double_delta(x, y, f=None, eps=1e-7):
    r"""Calculate double-delta integrals via 2D tetrahedron method .

    .. math::
        I(z) = \frac 1 N \sum_{\vec k}
            \delta(x_{\vec k} - z) \delta(y_{\vec k} - z) f_{\vec k}

    Parameters
    ----------
    x, y, f : ndarray
        Three functions sampled on uniform N x N mesh.
    eps : ndarray
        Negligible difference between fractional mesh-point indices.

    Returns
    -------
    function : float -> dict
        Intersection points of :math:`x, y = z` isolines and corresponding
        weights as a function of :math:`z`. The above double-delta integral
        :math:`I(z)` can be calculated as:

        .. code-block:: python

            I_z = sum(double_delta(x, y, f)(z).values())
    """
    N, N = x.shape

    if f is None:
        f = np.ones((N, N), dtype=int)

    triangles = [np.array([(i + k, j + k), (i + 1, j), (i, j + 1)])
        for i in range(N)
        for j in range(N)
        for k in range(2)
        if comm.rank == ((i * N + j) * 2 + k) % comm.size
        ]

    indices = [tuple(v.T % N) for v in triangles]

    triangles = [(v, x[i], y[i], f[i]) for v, i in zip(triangles, indices)]

    prefactor = 1.0 / N ** 2

    def dd(z):
        my_D = []
        my_W = []

        for (X, Y, Z), (A, B, C), (a, b, c), (F, G, H) in triangles:
            w = A * b - B * a - A * c + C * a + B * c - C * b
            # = sum[ijk] epsilon(ijk) F(i) f(j)

            if w == 0:
                continue

            U = (z * b - B * z - z * c + C * z + B * c - C * b) / w # A = a = z
            V = (A * z - z * a - A * c + C * a + z * c - C * z) / w # B = b = z
            W = (A * b - B * a - A * z + z * a + B * z - z * b) / w # C = c = z

            if 0 <= U <= 1 and 0 <= V <= 1 and 0 <= W <= 1:
                my_D.append(U * X + V * Y + W * Z)
                my_W.append(prefactor * (U * F + V * G + W * H) / abs(w))

        sizes = np.array(comm.allgather(len(my_W)))
        size = sizes.sum()

        D = np.empty((size, 2))
        W = np.empty(size)

        comm.Gatherv(np.array(my_D), (D, sizes * 2))
        comm.Gatherv(np.array(my_W), (W, sizes))

        unique = dict()

        if comm.rank == 0:
            for group in misc.group(D, eps=eps):
                d = np.average(D[group], axis=0)
                w = np.average(W[group], axis=0)
                unique[tuple(d)] = w

        return comm.bcast(unique)

    return dd

def isoline(energies):
    r"""Calculate isoline on triangular mesh (2D tetrahedron method)."""

    N, N = energies.shape

    triangles = set()

    dk1 = [1, 0, -1, -1, 0, 1]
    dk2 = [0, 1, 1, 0, -1, -1]

    fun = bravais.linear_interpolation(energies)

    for k1 in range(N):
        for k2 in range(N):
            images = bravais.to_Voronoi(k1, k2, N)

            if len(images) == 1:
                K1, K2 = images.pop()

                for n in range(-1, 5):
                    triangle = [
                        (K1, K2),
                        (K1 + dk1[n], K2 + dk2[n]),
                        (K1 + dk1[n + 1], K2 + dk2[n + 1]),
                        ]

                    for n, (c1, c2) in enumerate(triangle):
                        if c1 - c2 > N:
                            triangle[n] = (c1 - 0.5, c2 + 0.5)
                        elif c1 - c2 < -N:
                            triangle[n] = (c1 + 0.5, c2 - 0.5)
                        elif 2 * c1 + c2 > N:
                            triangle[n] = (c1 - 0.5, c2)
                        elif 2 * c1 + c2 < -N:
                            triangle[n] = (c1 + 0.5, c2)
                        elif c1 + 2 * c2 > N:
                            triangle[n] = (c1, c2 - 0.5)
                        elif c1 + 2 * c2 < -N:
                            triangle[n] = (c1, c2 + 0.5)

                    triangles.add(tuple(sorted(triangle,
                        key=lambda x: (fun(*x), x))))

    triangles = list(triangles)
    triangles = triangles[comm.rank::comm.size]

    triangles = [(np.array(v), fun(*zip(*v))) for v in triangles]

    def FS(E):
        my_points = []

        for (i, j, k), (A, B, C) in triangles:
            if A < E <= B:
                if E == B == C:
                    my_points.append((tuple(j), tuple(k)))
                else:
                    alpha = (E - A) / (B - A)
                    beta = (E - A) / (C - A)
                    my_points.append((
                        tuple(i * (1 - alpha) + j * alpha),
                        tuple(i * (1 - beta) + k * beta),
                        ))

            elif B <= E < C:
                if E == A == B:
                    my_points.extend((tuple(i), (j)))
                else:
                    alpha = (E - B) / (C - B)
                    beta = (E - A) / (C - A)
                    my_points.append((
                        tuple(j * (1 - alpha) + k * alpha),
                        tuple(i * (1 - beta) + k * beta),
                        ))

        points = set()

        for group in comm.allgather(my_points):
            points.update(group)

        try:
            contours = [list(points.pop())]
        except KeyError:
            return []

        while points:
            for point in points:
                if contours[-1][-1] in point:
                    contours[-1].append(
                        point[1 - point.index(contours[-1][-1])])
                    points.remove(point)
                    break
                elif contours[-1][0] in point:
                    contours[-1].insert(0,
                        point[1 - point.index(contours[-1][0])])
                    points.remove(point)
                    break
            else:
                contours.append(list(points.pop()))

        return [np.array(contour) / N for contour in contours]

    return FS
