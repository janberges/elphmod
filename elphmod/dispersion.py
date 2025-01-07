# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

"""Diagonalization on paths and meshes."""

import numpy as np

import elphmod.bravais
import elphmod.misc
import elphmod.MPI

comm = elphmod.MPI.comm

def dispersion(matrix, k, angle=60, vectors=False, gauge=False, rotate=False,
        order=False, hermitian=True, broadcast=True, shared_memory=False,
        **order_kwargs):
    r"""Diagonalize Hamiltonian or dynamical matrix for given k points.

    Parameters
    ----------
    matrix : function
        Matrix to be diagonalized as a function of k in crystal coordinates with
        period :math:`2 \pi`.
    k : ndarray
        k points in crystal coordinates with period :math:`2 \pi`. Wave-vector
        components are stored along the last axis; the leading axes describe
        the mesh. The shapes of `matrix` and `k` together determine the shape
        of the results.
    angle : float
        Angle between the axes of the reciprocal lattice.
    vectors : bool
        Return eigenvectors?
    gauge : bool
        Choose largest element of each eigenvector to be real? Not stable!
    rotate : bool
        Align (phonon) eigenvectors with wave vector k via in-plane rotation.
        This is experimental and supposed to support the band-order algorithm.
    order : bool
        Order/disentangle bands via their k-local character? Depending on the
        topology of the band structure, this may not be possible. Adjacent
        points in `k` must be adjacent in the Brillouin zone too.
    hermitian : bool
        Assume `matrix` to be Hermitian, i.e., calculate real eigenvalues?
    broadcast : bool
        Broadcast result from rank 0 to all processes?
    shared_memory : bool
        Store results in shared memory?
    **order_kwargs
        Keyword arguments passed to :func:`band_order`.

    Returns
    -------
    ndarray
        Eigenvalues for the given k points.
    ndarray, optional
        Corresponding eigenvectors.
    ndarray, optional
        Indices which have been used to order the bands.
    """
    if comm.rank == 0:
        k = np.array(k)
        kshape = k.shape[:max(1, k.ndim - 1)]
        points = np.prod(kshape)
        k = np.reshape(k, (points, -1))
        dimens = k.shape[-1]
    else:
        kshape = points = dimens = None

    kshape = comm.bcast(kshape) # k-mesh dimensions
    points = comm.bcast(points) # number of k points
    dimens = comm.bcast(dimens) # number of dimensions

    bands = int(round(np.sqrt(matrix().size))) # number of bands

    # choose number of k points to be processed by each processor:

    my_points = elphmod.MPI.distribute(points)

    # initialize local lists of k points, eigenvalues and eigenvectors:

    my_k = np.empty((my_points[comm.rank], dimens))
    my_v = np.empty((my_points[comm.rank], bands),
        dtype=float if hermitian else complex)

    if order or vectors:
        my_V = np.empty((my_points[comm.rank], bands, bands), dtype=complex)

    # distribute k points among processors:

    comm.Scatterv((k, my_points * dimens), my_k)

    # diagonalize matrix for local lists of k points:

    status = elphmod.misc.StatusBar(my_points[comm.rank],
        title='calculate dispersion')

    if rotate:
        a1, a2 = elphmod.bravais.translations(180 - angle)
        b1, b2 = elphmod.bravais.reciprocals(a1, a2)

    for point in range(len(my_k)):
        matrix_k = matrix(*my_k[point])

        if order or vectors:
            if bands == 1:
                my_v[point] = matrix_k.real if hermitian else matrix_k
                my_V[point] = 1.0
            elif hermitian:
                my_v[point], my_V[point] = np.linalg.eigh(matrix_k)
            else:
                my_v[point], my_V[point] = np.linalg.eig(matrix_k)

                my_order = np.argsort(my_v[point])

                my_v[point] = my_v[point][my_order]
                my_V[point] = my_V[point][:, my_order]

            if gauge:
                for band in range(bands):
                    my_V[point, :, band] *= np.exp(-1j * np.angle(
                        max(my_V[point, :, band], key=abs)))

            # rotate phonon eigenvectors by negative angle of k point:

            if rotate:
                k1, k2 = elphmod.bravais.to_Voronoi(*my_k[point, :2],
                    nk=2 * np.pi, angle=angle)[0]

                x, y = k1 * b1 + k2 * b2
                phi = np.arctan2(y, x)

                atoms = bands // 3

                for atom in range(atoms):
                    for band in range(bands):
                        xy = point, [atom, atom + atoms], band
                        my_V[xy] = elphmod.bravais.rotate(my_V[xy], -phi)
        else:
            if bands == 1:
                my_v[point] = matrix_k.real if hermitian else matrix_k
            elif hermitian:
                my_v[point] = np.linalg.eigvalsh(matrix_k)
            else:
                my_v[point] = np.linalg.eigvals(matrix_k)
                my_v[point].sort()

        status.update()

    # gather calculated eigenvectors on first processor:

    memory = dict(shared_memory=shared_memory, single_memory=not broadcast)

    node, images, v = elphmod.MPI.shared_array((points, bands),
        dtype=my_v.dtype, **memory)

    comm.Gatherv(my_v, (v, my_points * bands))

    if order or vectors:
        node, images, V = elphmod.MPI.shared_array((points, bands, bands),
            dtype=complex, **memory)

        comm.Gatherv(my_V, (V, my_points * bands ** 2))

    # order/disentangle bands:

    if order:
        node, images, o = elphmod.MPI.shared_array((points, bands),
            dtype=int, **memory)

        if comm.rank == 0:
            o = band_order(v, V, **order_kwargs)

            for point in range(points):
                v[point] = v[point, o[point]]

                if vectors:
                    for band in range(bands):
                        V[point, band] = V[point, band, o[point]]

    # broadcast results:

    if broadcast:
        if node.rank == 0:
            images.Bcast(v)

            if vectors:
                images.Bcast(V)

            if order:
                images.Bcast(o)

        node.Barrier()

    # reshape results:

    if broadcast or comm.rank == 0:
        v = np.reshape(v, (*kshape, bands))

        if vectors:
            V = np.reshape(V, (*kshape, bands, bands))

        if order:
            o = np.reshape(o, (*kshape, bands))

    # return results:

    if vectors and order:
        return v, V, o

    if vectors:
        return v, V

    if order:
        return v, o

    return v

def dispersion_full(matrix, size, angle=60, vectors=False, gauge=False,
        rotate=False, order=False, hermitian=True, broadcast=True,
        shared_memory=False, **order_kwargs):
    """Diagonalize Hamiltonian or dynamical matrix on uniform k-point mesh."""

    # choose irreducible set of k points:

    k = np.array(sorted(elphmod.bravais.irreducibles(size, angle=angle)))

    points = len(k) # number of k points

    bands = int(round(np.sqrt(matrix().size))) # number of bands

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
            main_order = band_order(v[main_path], V[main_path], status=False,
                **order_kwargs)

            status = elphmod.misc.StatusBar(points, title='disentangle bands')

            for n, N in zip(main_path, main_order):
                side_path = [m for m in range(points) if on_side_path(n, m)]
                side_order = band_order(v[side_path], V[side_path],
                    by_mean=False, status=False, **order_kwargs)

                for m, M in zip(side_path, side_order):
                    o[m] = M[N]
                    v[m] = v[m, o[m]]

                    if vectors:
                        for band in range(bands):
                            V[m, band] = V[m, band, o[m]]

                    status.update()

    else:
        v = dispersion(matrix, 2 * np.pi / size * k, angle=angle,
            vectors=False, gauge=False, rotate=False, order=False,
            hermitian=hermitian, broadcast=True)

    # fill uniform mesh with data from irreducible wedge:

    memory = dict(shared_memory=shared_memory, single_memory=not broadcast)

    node, images, v_mesh = elphmod.MPI.shared_array((size, size, bands),
        dtype=v.dtype, **memory)

    if vectors:
        node, images, V_mesh = elphmod.MPI.shared_array(
            (size, size, bands, bands), dtype=complex, **memory)

    if order:
        node, images, o_mesh = elphmod.MPI.shared_array((size, size, bands),
            dtype=int, **memory)

    if comm.rank == 0:
        # transfer data points from wedge to mesh:

        for point, (k1, k2) in enumerate(k):
            for K1, K2 in elphmod.bravais.images(k1, k2, size, angle=angle):
                v_mesh[K1, K2] = v[point]

                if vectors:
                    V_mesh[K1, K2] = V[point]

                if order:
                    o_mesh[K1, K2] = o[point]

    # broadcast results:

    if broadcast:
        if node.rank == 0:
            images.Bcast(v_mesh)

            if vectors:
                images.Bcast(V_mesh)

            if order:
                images.Bcast(o_mesh)

        node.Barrier()

    if vectors and order:
        return v_mesh, V_mesh, o_mesh

    if vectors:
        return v_mesh, V_mesh

    if order:
        return v_mesh, o_mesh

    return v_mesh

def dispersion_full_nosym(matrix, size, vectors=False, order=False,
        *args, **kwargs):
    """Diagonalize Hamiltonian or dynamical matrix on uniform k-point mesh.

    Use this routine to get eigenvectors less symmetric than the eigenvalues!
    """
    # set up k-point mesh:

    if comm.rank == 0:
        if hasattr(size, '__iter__') and not order:
            k = elphmod.bravais.mesh(*size)
        else:
            k = elphmod.bravais.mesh(size, size)
    else:
        k = None

    # calculate dispersion:

    v = dispersion(matrix, k, vectors=vectors, order=False, *args, **kwargs)

    if vectors:
        v, V = v

    # order/disentangle bands:

    if order:
        _, o = dispersion_full(matrix, size, vectors=False, order=True,
            *args, **kwargs)

        for k1 in range(size):
            for k2 in range(size):
                v[k1, k2] = v[k1, k2, o[k1, k2]]

                if vectors:
                    for n in range(V.shape[2]):
                        V[k1, k2, n] = V[k1, k2, n, o[k1, k2]]

    # return results:

    if vectors and order:
        return v, V, o

    if vectors:
        return v, V

    if order:
        return v, o

    return v

def sample(matrix, k, **kwargs):
    """Calculate Hamiltonian or dynamical matrix for given k points.

    Parameters
    ----------
    matrix : function
        Matrix as a function of k.
    k : ndarray
        k points.
    **kwargs
        Keyword arguments passed to `matrix`.
    """
    k = np.array(k)
    kshape = k.shape[:max(1, k.ndim - 1)]
    k = np.reshape(k, (np.prod(kshape), -1))

    sizes, bounds = elphmod.MPI.distribute(len(k), bounds=True)

    template = matrix(**kwargs)

    my_matrix = np.empty((sizes[comm.rank], *template.shape),
        dtype=template.dtype)

    status = elphmod.misc.StatusBar(sizes[comm.rank], title='sample something')

    for my_ik, ik in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        my_matrix[my_ik] = matrix(*k[ik], **kwargs)

        status.update()

    matrix = np.empty((*kshape, *template.shape), dtype=template.dtype)

    comm.Allgatherv(my_matrix, (matrix, sizes * template.size))

    return matrix

def band_order(v, V, by_mean=True, dv=float('inf'), eps=1e-10, status=True):
    """Sort bands by overlap of eigenvectors at neighboring k points.

    Parameters
    ----------
    v : ndarray
        List of eigenvalues.
    V : ndarray
        Corresponding list of eigenvectors.
    by_mean : bool
        Sort disentangled bands by average frequency?
    dv : float
        Maximum allowed difference between consecutive eigenvalues of the same
        band after disentanglement.
    eps : float
        Maximum difference between eigenvalues considered degenerate.
    status : bool
        Show progress bar?

    Returns
    -------
    ndarray
        List of band indices that sort eigenstates.
    """
    points, bands = v.shape

    o = np.empty((points, bands), dtype=int)

    n0 = 0
    o[n0] = range(bands)

    if status:
        bar = elphmod.misc.StatusBar(points - 1, title='disentangle bands')

    for n in range(1, points):
        available = set(range(bands))

        for i in range(bands):
            o[n, i] = max(available, key=lambda j:
                abs(np.dot(V[n0, :, o[n0, i]], V[n, :, j].conj()))
                if abs(v[n0, o[n0, i]] - v[n, j]) < dv else 0)

            available.remove(o[n, i])

        # Only eigenvectors belonging to different eigenvalues are guaranteed to
        # be orthogonal. Thus k points with degenerate eigenvectors are not used
        # as starting point:

        if np.all(np.absolute(np.diff(v[n])) > eps):
            n0 = n

        if status:
            bar.update()

    if by_mean:
        o[:] = o[:, sorted(range(bands), key=lambda i: v[:, o[:, i]].sum())]

    return o

def map_dispersions(V1, V2):
    """Map two similar arrays of eigenvectors onto each other."""

    shape = V1.shape

    points = np.prod(shape[:-2], dtype=int)
    bands = shape[-2]

    V1 = np.reshape(V1, (points, bands, bands))
    V2 = np.reshape(V2, (points, bands, bands))

    mapping = np.empty((points, bands), dtype=int)

    for n in range(points):
        available = set(range(bands))

        for i in range(bands):
            mapping[n, i] = max(available, key=lambda j:
                np.absolute(np.dot(V1[n, :, i], V2[n, :, j].conj())))

            available.remove(mapping[n, i])

    return np.reshape(mapping, shape[:-1])

def unfolding_weights(k, R, U0, U, blocks0=None, blocks=None, sgn=-1):
    r"""Calculate weights for "unfolding" of supercell dispersions.

    Parameters
    ----------
    k : list of d-tuples
        k points in arbitrary representation.
    R : list of d-tuples
        Positions of the unit cells in the supercell. The representation must
        be compatible with the k points: If `k` is given in crystal coordinates
        with a period of :math:`2 \pi`, `R` must be given in crystal
        coordinates with a period of 1. `k` and `R` can also be defined in
        Cartesian coordinates.
    U0: ndarray
        Eigenvectors of the symmetric system.
    U: ndarray
        Eigenvectors of the supercell system.
    blocks0 : list of indexing objects, optional
        Mapping from indices of `R` to slices of `U0`. By default, all orbitals
        are selected.
    blocks : list of indexing objects, optional
        Mapping from indices of `R` to slices of `U`. By default, it is assumed
        that the orbitals are grouped into blocks corresponding to unit cells.
        Within the blocks, the order of the orbitals is as in `U0`. The order
        of the blocks is as in `R`.
    sgn : int
        Sign convention for Fourier transform in tight-binding model.
        The default sign ``-1`` is suitable for data from Wannier90 as provided
        by :meth:`elphmod.el.Model.H`.
        `Other conventions <https://doi.org/10.26092/elib/250>`_ require ``+1``.

    Returns
    -------
    ndarray
        Weights of the supercell states.
    """
    bands0 = U0.shape[-1]
    bands = U.shape[-1]

    if blocks0 is None:
        blocks0 = [slice(U0.shape[1])] * len(R)

    if blocks is None:
        blocks = [slice(ir * U0.shape[1], (ir + 1) * U0.shape[1])
            for ir in range(len(R))]

    U0 = U0 / np.sqrt(bands / bands0)

    sizes, bounds = elphmod.MPI.distribute(len(k), bounds=True)

    my_w = np.empty((sizes[comm.rank], bands))

    status = elphmod.misc.StatusBar(sizes[comm.rank], title='unfold bands')

    for my_ik, ik in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        my_w[my_ik] = sum(abs(sum(np.exp(sgn * 1j * np.dot(k[ik], R[ir]))
            * np.dot(U0[ik, blocks0[ir], m].conj(), U[ik, blocks[ir]])
                for ir in range(len(R)))) ** 2
                for m in range(bands0))

        status.update()

    w = np.empty((len(k), bands))

    comm.Allgatherv(my_w, (w, sizes * bands))

    return w
