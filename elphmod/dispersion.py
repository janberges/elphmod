#/usr/bin/env python

import numpy as np
import numpy.linalg

from . import bravais, MPI
comm = MPI.comm

def dispersion(matrix, k, angle=60, vectors=False, gauge=False, rotate=False,
        order=False, hermitian=True, broadcast=True):
    """Diagonalize Hamiltonian or dynamical matrix for given k points."""

    points = len(k) if comm.rank == 0 else None
    points = comm.bcast(points) # number of k points

    bands = matrix.size # number of bands

    # choose number of k points to be processed by each processor:

    my_points = MPI.distribute(points)

    # initialize local lists of k points, eigenvalues and eigenvectors:

    my_k = np.empty((my_points[comm.rank], 2))
    my_v = np.empty((my_points[comm.rank], bands))

    if order or vectors:
        my_V = np.empty((my_points[comm.rank], bands, bands), dtype=complex)

    # distribute k points among processors:

    comm.Scatterv((k, my_points * 2), my_k)

    # diagonalize matrix for local lists of k points:

    if rotate:
        t1, t2 = bravais.translations(180 - angle)
        u1, u2 = bravais.reciprocals(t1, t2)

    for point, (k1, k2) in enumerate(my_k):
        matrix_k = matrix(k1, k2)

        if order or vectors:
            if bands == 1:
                my_v[point], my_V[point] = matrix_k.real, 1
            elif hermitian:
                my_v[point], my_V[point] = np.linalg.eigh(matrix_k)
            else:
                tmp, my_V[point] = np.linalg.eig(matrix_k)
                my_v[point] = tmp.real

                tmp = np.argsort(my_v[point])

                my_v[point] = my_v[point][tmp]
                my_V[point] = my_V[point][:, tmp]

            if gauge:
                for band in range(bands):
                    my_V[point, :, band] *= np.exp(-1j * np.angle(
                        max(my_V[point, :, band], key=abs)))

            # rotate phonon eigenvectors by negative angle of k point:

            if rotate:
                K1, K2 = bravais.to_Voronoi(k1, k2, 2 * np.pi, angle=angle)[0]

                x, y = K1 * u1 + K2 * u2
                phi = np.arctan2(y, x)

                atoms = bands // 3

                for atom in range(atoms):
                    for band in range(bands):
                        xy = point, [atom, atom + atoms], band
                        my_V[xy] = bravais.rotate(my_V[xy], -phi)
        else:
            if bands == 1:
                my_v[point] = matrix_k.real
            elif hermitian:
                my_v[point] = np.linalg.eigvalsh(matrix_k)
            else:
                my_v[point] = np.linalg.eigvals(matrix_k).real.sort()

    # gather calculated eigenvectors on first processor:

    v = np.empty((points, bands))
    comm.Gatherv(my_v, (v, my_points * bands))

    if order or vectors:
        V = np.empty((points, bands, bands), dtype=complex)
        comm.Gatherv(my_V, (V, my_points * bands ** 2))

    # order/disentangle bands:

    if order:
        o = np.empty((points, bands), dtype=int)

        if comm.rank == 0:
            o = band_order(v, V)

            for point in range(points):
                v[point] = v[point, o[point]]

                if vectors:
                    for band in range(bands):
                        V[point, band] = V[point, band, o[point]]

    # broadcast results:

    if broadcast:
        comm.Bcast(v)

        if vectors:
            comm.Bcast(V)

        if order:
            comm.Bcast(o)

    if vectors and order:
        return v, V, o

    if vectors:
        return v, V

    if order:
        return v, o

    return v

def dispersion_full(matrix, size, angle=60, vectors=False, gauge=False,
        rotate=False, order=False, hermitian=True, broadcast=True):
    """Diagonalize Hamiltonian or dynamical matrix on uniform k-point mesh."""

    # choose irreducible set of k points:

    k = np.array(sorted(bravais.irreducibles(size, angle=angle)))

    points = len(k)      # number of k points
    bands  = matrix.size # number of bands

    # define main and side paths for different axes:

    def on_main_path(n):
        return not k[n, 0]

    if angle == 60 or angle == 90:
        def on_side_path(n, m):
            return k[m, 1] == k[n, 1]
    elif angle == 120:
        def on_side_path(n, m):
            return k[m, 1] - k[m, 0] == k[n, 1] - k[n, 0]

    # calculate dispersion using the above routine:

    if order or vectors:
        v, V = dispersion(matrix, 2 * np.pi / size * k, angle=angle,
            vectors=True, gauge=gauge, rotate=rotate, order=False,
            hermitian=hermitian, broadcast=False)

        # order bands along spider-web-like paths:
        #
        # example: irreducible wedge of 12 x 12 mesh
        # ------------------------------------------
        #
        # hexagonal lattice       K      G = (0 0)
        # -----------------      /       M = (0 6)
        #                   o   o        K = (4 4) for angle =  60
        #                  /   /           = (4 8) for angle = 120
        #             o   o   o   o
        #            /   /   /   /    <- side paths to K
        #       o   o   o   o   o
        #      /   /   /   /   /
        # G---o---o---o---o---o---M   <- main path from G to M
        #
        # square lattice          M
        # --------------          |
        #                     o   o
        #                     |   |
        #                 o   o   o      G = (0 0)
        #                 |   |   |      X = (0 6)
        #             o   o   o   o      M = (6 6)
        #             |   |   |   |
        #         o   o   o   o   o
        #         |   |   |   |   |   <- side paths to M
        #     o   o   o   o   o   o
        #     |   |   |   |   |   |
        # G---o---o---o---o---o---X   <- main path from G to X

        if order and comm.rank == 0:
            o = np.empty((points, bands), dtype=int)

            main_path = [n for n in range(points) if on_main_path(n)]
            main_order = band_order(v[main_path], V[main_path])

            for n, N in zip(main_path, main_order):
                side_path = [m for m in range(points) if on_side_path(n, m)]
                side_order = band_order(v[side_path], V[side_path],
                    by_mean=False)

                for m, M in zip(side_path, side_order):
                    o[m] = M[N]
                    v[m] = v[m, o[m]]

                    if vectors:
                        V[m] = V[m, :, o[m]]

    else:
        v = dispersion(matrix, 2 * np.pi / size * k, angle=angle,
            vectors=False, gauge=False, rotate=False, order=False,
            hermitian=hermitian, broadcast=True)

    # fill uniform mesh with data from irreducible wedge:

    v_mesh = np.empty((size, size, bands))

    if vectors:
        V_mesh = np.empty((size, size, bands, bands), dtype=complex)

    if order:
        o_mesh = np.empty((size, size, bands), dtype=int)

    if comm.rank == 0:
        # transfer data points from wedge to mesh:

        for point, (k1, k2) in enumerate(k):
            for K1, K2 in bravais.images(k1, k2, size, angle=angle):
                v_mesh[K1, K2] = v[point]

                if vectors:
                    V_mesh[K1, K2] = V[point]

                if order:
                    o_mesh[K1, K2] = o[point]

    # broadcast results:

    if broadcast:
        comm.Bcast(v_mesh)

        if vectors:
            comm.Bcast(V_mesh)

        if order:
            comm.Bcast(o_mesh)

    if vectors and order:
        return v_mesh, V_mesh, o_mesh

    if vectors:
        return v_mesh, V_mesh

    if order:
        return v_mesh, o_mesh

    return v_mesh

def dispersion_full_nosym(matrix, size, *args, **kwargs):
    """Diagonalize Hamiltonian or dynamical matrix on uniform k-point mesh.

    Use this routine to get eigenvectors less symmetric than the eigenvalues!"""

    if comm.rank == 0:
        k = np.empty((size * size, 2))

        n = 0
        for k1 in range(size):
            for k2 in range(size):
                k[n] = k1, k2
                n += 1

        k *= 2 * np.pi / size
    else:
        k = None

    out = list(dispersion(matrix, k, *args, **kwargs))

    for n in range(len(out)):
        out[n] = np.reshape(out[n], (size, size) + out[n].shape[1:])

    return out

def band_order(v, V, by_mean=True):
    # dv=float('inf')

    """Sort bands by overlap of eigenvectors at neighboring k points."""

    points, bands = v.shape

    o = np.empty((points, bands), dtype=int)

    n0 = 0
    o[n0] = range(bands)

    for n in range(1, points):
        available = set(range(bands))

        for i in range(bands):
            o[n, i] = max(available, key=lambda j:
                np.absolute(np.dot(V[n0, :, o[n0, i]], V[n, :, j].conj())))

            available.remove(o[n, i])

            #o[n, i] = max(range(bands), key=lambda j:
            #       abs(np.dot(V[n0, :, o[n0, i]],  V[n, :, j].conj()))
            #    if abs(       v[n0,    o[n0, i]] - v[n,    j]) < dv else -1)

        # Only eigenvectors belonging to different eigenvalues are guaranteed to
        # be orthogonal. Thus k points with degenerate eigenvectors are not used
        # as starting point:

        if np.all(np.absolute(np.diff(v[n])) > 1e-10):
            n0 = n

    # reorder disentangled bands by average frequency:

    if by_mean:
        o[:] = o[:, sorted(range(bands), key=lambda i: v[:, o[:, i]].sum())]

    return o

def map_dispersions(V1, V2):
    """Map two similar arrays of eigenvectors onto each other."""

    shape = V1.shape

    points = np.prod(shape[:-2])
    bands = shape[-2]

    V1 = np.reshape(V1, (points, bands, bands))
    V2 = np.reshape(V2, (points, bands, bands))

    mapping = np.empty((points, bands), dtype=int)

    for n in range(points):
        for i in range(bands):
            mapping[n, i] = max(range(bands), key=lambda j:
                np.absolute(np.dot(V1[n, :, i], V2[n, :, j].conj())))

    return np.reshape(mapping, shape[:-1])
