# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

"""Lattices, symmetries, and interpolation."""

import numpy as np

import elphmod.misc
import elphmod.MPI

comm = elphmod.MPI.comm
info = elphmod.MPI.info

deg = np.pi / 180

def rotate(vector, angle, two_dimensional=True):
    """Rotate vector anti-clockwise.

    Parameters
    ----------
    vector : array_like
        Two-dimensional vector.
    angle : float
        Rotation angle in radians.

    Returns
    -------
    ndarray
        Rotated vector.
    """
    cos = np.cos(angle)
    sin = np.sin(angle)

    if two_dimensional:
        rotation = np.array([
            [cos, -sin],
            [sin,  cos],
        ])

        return np.dot(rotation, vector)
    else:
        rotation = np.array([
            [cos, -sin, 0.0],
            [sin,  cos, 0.0],
            [0.0,  0.0, 1.0],
        ])

        return np.dot(rotation, vector)

def primitives(ibrav=8, a=1.0, b=1.0, c=1.0, cosbc=0.0, cosac=0.0, cosab=0.0,
        celldm=None, bohr=False, r_cell=None, cell_units=None, **ignore):
    """Get primitive vectors of Bravais lattice as in QE.

    Adapted from Modules/latgen.f90 of Quantum ESPRESSO.

    For documentation, see http://www.quantum-espresso.org/Doc/INPUT_PW.html.

    Parameters
    ----------
    ibrav : int
        Bravais-lattice index.
    a, b, c, cosbc, cosac, cosab : float
        Traditional crystallographic constants in angstrom.
    celldm : list of float
        Alternative crystallographic constants. The first element is the
        lattice constant in bohr; the other elements are dimensionless.
    bohr : bool, default False
        Return lattice vectors in angstrom or bohr?
    r_cell, cell_units
        Cell parameters from 'func'`read_pwi` used if `ibrav` is zero.
    **ignore
        Ignored keyword arguments, e.g., parameters from 'func'`read_pwi`.

    Returns
    -------
    ndarray
        Matrix of primitive Bravais lattice vectors.
    """
    if celldm is None:
        celldm = np.zeros(6)

        celldm[0] = a / elphmod.misc.a0
        celldm[1] = b / a
        celldm[2] = c / a

        if ibrav in {0, 14}:
            celldm[3] = cosbc
            celldm[4] = cosac
            celldm[5] = cosab

        elif ibrav in {-13, -12}:
            celldm[4] = cosac

        elif ibrav in {-5, 5, 12, 13}:
            celldm[3] = cosab

    if not bohr:
        celldm[0] *= elphmod.misc.a0

    if ibrav == 0: # free
        if r_cell is None or cell_units is None:
            info('ibrav=0 requires r_cell and cell_units!', error=True)

        a = np.array(r_cell)

        if 'alat' in cell_units.lower():
            a *= celldm[0]
        elif bohr and 'angstrom' in cell_units.lower():
            a /= elphmod.misc.a0
        elif not bohr and 'bohr' in cell_units.lower():
            a *= elphmod.misc.a0

        return a

    if ibrav == 1: # cubic (sc)
        return np.eye(3) * celldm[0]

    if ibrav == 2: # cubic (fcc)
        return np.array([
            [-1.0, 0.0, 1.0],
            [ 0.0, 1.0, 1.0],
            [-1.0, 1.0, 0.0],
        ]) * celldm[0] / 2

    if ibrav == 3: # cubic (bcc)
        return np.array([
            [ 1.0,  1.0, 1.0],
            [-1.0,  1.0, 1.0],
            [-1.0, -1.0, 1.0],
        ]) * celldm[0] / 2

    if ibrav == -3: # cubic (bcc, more symmetric axis)
        return np.array([
            [-1.0,  1.0,  1.0],
            [ 1.0, -1.0,  1.0],
            [ 1.0,  1.0, -1.0],
        ]) * celldm[0] / 2

    if ibrav == 4: # hexagonal & trigonal
        return np.array([
            [ 1.0,              0.0,       0.0],
            [-0.5, 0.5 * np.sqrt(3),       0.0],
            [ 0.0,              0.0, celldm[2]],
        ]) * celldm[0]

    if ibrav == 5: # trigonal (3-fold axis c)
        tx = np.sqrt((1 - celldm[3]) / 2)
        ty = np.sqrt((1 - celldm[3]) / 6)
        tz = np.sqrt((1 + 2 * celldm[3]) / 3)

        return np.array([
            [ tx,    -ty, tz],
            [  0, 2 * ty, tz],
            [-tx,    -ty, tz],
        ]) * celldm[0]

    if ibrav == -5: # trigonal (3-fold axis <111>)
        tx = np.sqrt((1 - celldm[3]) / 2)
        ty = np.sqrt((1 - celldm[3]) / 6)
        tz = np.sqrt((1 + 2 * celldm[3]) / 3)

        u = tz - 2 * np.sqrt(2) * ty
        v = tz + np.sqrt(2) * ty

        return np.array([
            [u, v, v],
            [v, u, v],
            [v, v, u],
        ]) * celldm[0] / np.sqrt(3)

    if ibrav == 6: # tetragonal (st)
        return np.array([
            [ 1.0, 0.0,       0.0],
            [ 0.0, 1.0,       0.0],
            [ 0.0, 0.0, celldm[2]],
        ]) * celldm[0]

    if ibrav == 7: # tetragonal (bct)
        return np.array([
            [ 1.0, -1.0, celldm[2]],
            [ 1.0,  1.0, celldm[2]],
            [-1.0, -1.0, celldm[2]],
        ]) * celldm[0] / 2

    if ibrav == 8: # orthorhombic
        return np.diag([1.0, celldm[1], celldm[2]]) * celldm[0]

    if ibrav == 9: # orthorhombic (bco)
        return np.array([
            [ 0.5, 0.5 * celldm[1],       0.0],
            [-0.5, 0.5 * celldm[1],       0.0],
            [ 0.0,             0.0, celldm[2]],
        ]) * celldm[0]

    if ibrav == -9: # orthorhombic (bco, alternate description)
        return np.array([
            [0.5, -0.5 * celldm[1],       0.0],
            [0.5,  0.5 * celldm[1],       0.0],
            [0.0,              0.0, celldm[2]],
        ]) * celldm[0]

    if ibrav == 91: # orthorhombic (A-type)
        return np.array([
            [1.0,             0.0,              0.0],
            [0.0, 0.5 * celldm[1], -0.5 * celldm[2]],
            [0.0, 0.5 * celldm[1],  0.5 * celldm[2]],
        ]) * celldm[0]

    if ibrav == 10: # orthorhombic (fco)
        return np.array([
            [1.0,       0.0, celldm[2]],
            [1.0, celldm[1],       0.0],
            [0.0, celldm[1], celldm[2]],
        ]) * celldm[0] / 2

    if ibrav == 11: # orthorhombic (bco)
        return np.array([
            [ 1.0,  celldm[1], celldm[2]],
            [-1.0,  celldm[1], celldm[2]],
            [-1.0, -celldm[1], celldm[2]],
        ]) * celldm[0] / 2

    if ibrav == 12: # monoclinic (unique axis c)
        cos = celldm[3]
        sin = np.sqrt(1.0 - cos ** 2)

        return np.array([
            [            1.0,             0.0,       0.0],
            [celldm[1] * cos, celldm[1] * sin,       0.0],
            [            0.0,             0.0, celldm[2]],
        ]) * celldm[0]

    if ibrav == -12: # monoclinic (unique axis b)
        cos = celldm[4]
        sin = np.sqrt(1.0 - cos ** 2)

        return np.array([
            [            1.0,       0.0,             0.0],
            [            0.0, celldm[1],             0.0],
            [celldm[2] * cos,       0.0, celldm[2] * sin],
        ]) * celldm[0]

    if ibrav == 13: # monoclinic (bcm, unique axis c)
        cos = celldm[3]
        sin = np.sqrt(1.0 - cos ** 2)

        return np.array([
            [            0.5,             0.0, -0.5 * celldm[2]],
            [celldm[1] * cos, celldm[1] * sin,              0.0],
            [            0.5,             0.0,  0.5 * celldm[2]],
        ]) * celldm[0]

    if ibrav == -13: # monoclinic (bcm, unique axis b)
        cos = celldm[4]
        sin = np.sqrt(1.0 - cos ** 2)

        return np.array([
            [            0.5, 0.5 * celldm[1],             0.0],
            [           -0.5, 0.5 * celldm[1],             0.0],
            [celldm[2] * cos,             0.0, celldm[2] * sin],
        ]) * celldm[0]

    if ibrav == 14: # triclinic
        cosc = celldm[5]
        sinc = np.sqrt(1.0 - cosc ** 2)

        cosa = celldm[3]
        cosb = celldm[4]

        ex1 = (cosa - cosb * cosc) / sinc
        ex2 = np.sqrt(1.0 + 2.0 * cosa * cosb * cosc
            - cosa ** 2 - cosb ** 2 - cosc ** 2) / sinc

        return np.array([
            [             1.0,              0.0,            0.0],
            [celldm[1] * cosc, celldm[1] * sinc,            0.0],
            [celldm[2] * cosb, celldm[2] * ex1, celldm[2] * ex2],
        ]) * celldm[0]

    info('Bravais lattice %s unknown!' % ibrav, error=True)

def translations(angle=120, angle0=0, two_dimensional=True):
    """Generate translation vectors of Bravais lattice.

    Parameters
    ----------
    angle : float
        Angle between first and second vector in degrees::

            VALUE  LATTICE
               60  hexagonal
               90  square
              120  hexagonal (ibrav = 4 in Quantum ESPRESSO)

    angle0 : float
        Angle between x axis and first vector in degrees.

    Returns
    -------
    ndarray, ndarray
        Translation vectors of Bravais lattice.
    """
    if two_dimensional:
        a1 = np.array([1.0, 0.0])

        a1 = rotate(a1, angle0 * deg)
        a2 = rotate(a1, angle * deg)

        return a1, a2
    else:
        a1 = np.array([1.0, 0.0, 0.0])

        a1 = rotate(a1, angle0 * deg, two_dimensional=False)
        a2 = rotate(a1, angle * deg, two_dimensional=False)

        return a1, a2

def reciprocals(a1, a2, a3=None):
    r"""Generate translation vectors of reciprocal lattice.

    Parameters
    ----------
    a1, a2 : ndarray
        Translation vectors of Bravais lattice.

    Returns
    -------
    ndarray, ndarray
        Translation vectors of reciprocal lattice (without :math:`2 \pi`).
    """
    if a3 is None:
        b1 = rotate(a2, -90 * deg)
        b2 = rotate(a1, +90 * deg)

        b1 /= np.dot(a1, b1)
        b2 /= np.dot(a2, b2)

        return b1, b2

    else:
        b1 = np.cross(a2, a3)
        b2 = np.cross(a3, a1)
        b3 = np.cross(a1, a2)

        b1 /= np.dot(a1, b1)
        b2 /= np.dot(a2, b2)
        b3 /= np.dot(a3, b3)

        return b1, b2, b3

def volume(a1, a2=None, a3=None):
    """Calculate unit-cell volume/area/length.

    Parameters
    ----------
    a1, a2, a3 : ndarray
        Primite lattice vectors.

    Returns
    -------
    float
        Unit-cell volume/area/length.
    """
    if a2 is None and a3 is None:
        return a1[0]

    if a3 is None:
        return abs(a1[0] * a2[1] - a1[1] * a2[0])

    return abs(np.dot(a1, np.cross(a2, a3)))

def supercell(N1=1, N2=1, N3=1):
    """Set up supercell.

    Parameters
    ----------
    N1, N2, N3 : tuple of int or int, default 1
        Supercell lattice vectors in units of primitive lattice vectors.
        Multiples of single primitive vector can be defined via a scalar
        integer, linear combinations via a 3-tuple of integers.

    Returns
    -------
    int
        Number of unit cells in supercell.
    tuple of ndarray
        Integer vectors spanning the supercell.
    tuple of ndarray
        Integer vectors spanning the reciprocal supercell.
    list of tuple
        Integer positions of the unit cells in the supercell.
    """
    if not hasattr(N1, '__len__'): N1 = (N1, 0, 0)
    if not hasattr(N2, '__len__'): N2 = (0, N2, 0)
    if not hasattr(N3, '__len__'): N3 = (0, 0, N3)

    N1 = np.array(N1)
    N2 = np.array(N2)
    N3 = np.array(N3)

    N = np.dot(N1, np.cross(N2, N3))

    B1 = np.sign(N) * np.cross(N2, N3)
    B2 = np.sign(N) * np.cross(N3, N1)
    B3 = np.sign(N) * np.cross(N1, N2)

    N = abs(N)

    cells = []

    if comm.rank == 0:
        corners = np.array([n1 * N1 + n2 * N2 + n3 * N3
            for n1 in range(2)
            for n2 in range(2)
            for n3 in range(2)])

        n1_lower, n2_lower, n3_lower = corners.min(axis=0)
        n1_upper, n2_upper, n3_upper = corners.max(axis=0) + 1

        for n1 in range(n1_lower, n1_upper):
            for n2 in range(n2_lower, n2_upper):
                for n3 in range(n3_lower, n3_upper):
                    R = (n1, n2, n3)

                    if (0 <= np.dot(R, B1) < N and
                        0 <= np.dot(R, B2) < N and
                        0 <= np.dot(R, B3) < N):

                        cells.append(R)

        assert len(cells) == N

    cells = comm.bcast(cells)

    return N, (N1, N2, N3), (B1, B2, B3), cells

def to_supercell(R, supercell):
    """Map lattice vector to supercell.

    Parameters
    ----------
    R : tuple of int
        Unit-cell lattice vector.
    supercell : tuple
        Supercell info returned by :func:`supercell`.

    Returns
    -------
    tuple of int
        Supercell lattice vector.
    int
        Index of unit cell within supercell.
    """
    N, (N1, N2, N3), (B1, B2, B3), cells = supercell

    R1, r1 = divmod(np.dot(R, B1), N)
    R2, r2 = divmod(np.dot(R, B2), N)
    R3, r3 = divmod(np.dot(R, B3), N)

    r = (r1 * N1 + r2 * N2 + r3 * N3) // N

    return (R1, R2, R3), cells.index(tuple(r))

def images(k1, k2, nk, angle=60):
    """Generate symmetry-equivalent k points.

    Parameters
    ----------
    k1, k2 : int
        Indices of point in uniform mesh.
    nk : int
        Number of mesh points per dimension.
    angle : float
        Angle between mesh axes in degrees.

    Returns
    -------
    set
        Mesh-point indices of all equivalent k points.
    """
    points = set()

    while True:
        # rotation:

        if angle == 60: # by 60 deg
            k1, k2 = -k2, k1 + k2
        elif angle == 90: # by 90 deg
            k1, k2 = -k2, k1
        elif angle == 120: # by 60 deg
            k1, k2 = k1 - k2, k1

        # mapping to [0, nk):

        k1 %= nk
        k2 %= nk

        # add point or break loop after full rotation:

        if (k1, k2) in points:
            break
        else:
            points.add((k1, k2))

    # reflection:

    points |= set((k2, k1) for k1, k2 in points)

    return points

def irreducibles(nk, angle=60):
    r"""Generate set of irreducible k points.

    Parameters
    ----------
    nk : int
        Number of mesh points per dimension.
    angle : float
        Angle between mesh axes in degrees.

    Returns
    -------
    set
        Mesh-point indices of irreducible k points.

        Of all equivalent points, the first occurrence in the sequence

        .. math::

            (0, 0), (0, 1), \dots, (0, n_k - 1), (1, 0), (1, 1), \dots

        is chosen. :func:`sorted` should yield the same irreducible q points as
        used by Quantum ESPRESSO's PHonon code and found in the file *fildyn0*.
    """
    # set up sequence as described above:

    points = [
        (k1, k2)
        for k1 in range(nk)
        for k2 in range(nk)]

    irreducible = set(points)

    # remove as many equivalent points as possible:

    for k in points:
        if k in irreducible:
            reducible = images(*k, nk=nk, angle=angle)
            reducible.discard(k)
            irreducible -= reducible

    return irreducible

def symmetries(data, epsilon=0.0, unity=True, angle=60):
    """Find symmetries of data on Monkhorst-Pack mesh.

    Parameters
    ----------
    data : ndarray
        Data on uniform k mesh.
    epsilon : float
        Maximum absolute difference of "equal" floats.
    unity : bool
        Return identity as first symmetry?
    angle : float
        Angle between mesh axes in degrees.

    Returns
    -------
    iterator
        All symmetries found are returned one after the other.

        Each symmetry is described by a Boolean ("reflection?") and a rotation
        angle in degrees, followed by a mapping between the k-point indices of
        the original and the transformed mesh.
    """
    a1, a2 = translations(180 - angle, angle0=0)
    b1, b2 = reciprocals(a1, a2)

    # a1 and b2 must point in x and y direction, respectively,
    # to make below reflection work properly.

    nk = len(data)

    def get_image(reflect, angle):
        image = np.empty((nk, nk, 2), dtype=int)

        for k1 in range(nk):
            for k2 in range(nk):
                # rotation in Cartesian coordinates:

                K = rotate(k1 * b1 + k2 * b2, angle * deg)

                # reflection across the ky axis:

                if reflect:
                    K[0] *= -1

                # transform to mesh-point indices in [0, nk):

                K1 = int(round(np.dot(K, a1))) % nk
                K2 = int(round(np.dot(K, a2))) % nk

                # discard this symmetry if it is not fulfilled by data:

                if abs(data[k1, k2] - data[K1, K2]) > epsilon:
                    return None

                # otherwise set another element of the k-point mapping:

                image[k1, k2] = (K1, K2)

        return image

    # generate iterator through symmetries:

    dangle = 90 if angle == 90 else 60

    for reflect in False, True:
        for angle in range(0, 360, dangle):
            if reflect or angle or unity:
                image = get_image(reflect, angle)

                if image is not None:
                    yield (reflect, angle), image

def complete(data, angle=60):
    """Complete data on Monkhorst-Pack mesh.

    Parameters
    ----------
    data : ndarray
        Incomplete data on uniform mesh. Missing values are represented by NaN.
    angle : float
        Angle between mesh axes in degrees.

    Returns
    -------
    ndarray
        Input data with missing values determined via the symmetries found.
    """
    irreducible = list(zip(*np.where(np.logical_not(np.isnan(data)))))

    # return if the data is already complete:

    if len(irreducible) == data.size:
        return

    # test which expected lattice symmetries are fulfilled and fill the gaps:

    for symmetry, image in symmetries(data, unity=False, angle=angle):
        for k in irreducible:
            data[tuple(image[k])] = data[k]

        if not np.isnan(data).any():
            return

def complete_k(wedge, nq):
    """Calculate k dependence for equivalent q points.

    Parameters
    ----------
    wedge : ndarray
        Data on irreducible q wedge and uniform k mesh.
    nq : int
        Number of q points per dimension.

    Returns
    -------
    ndarray
        Data on uniform q and k meshes.
    """
    q = sorted(irreducibles(nq))

    nQ, nk, nk = wedge.shape

    mesh = np.empty((nq, nq, nk, nk), dtype=wedge.dtype)

    symmetries_q = [image
        for name, image in symmetries(np.zeros((nq, nq)), unity=True)]

    symmetries_k = [image
        for name, image in symmetries(np.zeros((nk, nk)), unity=True)]

    done = set()

    for sym_q, sym_k in zip(symmetries_q, symmetries_k):
        for iq, (q1, q2) in enumerate(q):
            Q1, Q2 = sym_q[q1, q2]

            if (Q1, Q2) in done:
                continue

            done.add((Q1, Q2))

            for k1 in range(nk):
                for k2 in range(nk):
                    K1, K2 = sym_k[k1, k2]

                    mesh[Q1, Q2, K1, K2] = wedge[iq, k1, k2]

    return mesh

def stack(*points, **kwargs):
    """Minimize distance of points on periodic axis via full-period shifts.

    Example:

    .. code-block:: python

        >>> stack(3, 5, 9, 12, period=10)
        [13, 15, 9, 12]

    .. code-block:: text

         In: ... | ox x   x| xo o   o| oo o   o| ...
        Out: ... | oo o   x| xx x   o| oo o   o| ...

    Parameters
    ----------
    *points
        Points on periodic axis.
    period : float
        Period of axis. Specified via `**kwargs` for Python-2 compatibility.

    Returns
    -------
    ndarray
        Points equivalent to input, with minimal distance on non-periodic axis.
    """
    period = kwargs.get('period', 2 * np.pi)

    # map points onto interval [0, period):

    points = np.array(points) % period

    # bring input into uniform shape:

    shape = points.shape

    points = points.reshape((points.shape[0], -1))

    # generate list of indices that would sort points:

    order = np.argsort(points, axis=0)

    # save "stacking", shift lowest point up by one period, and repeat:

    stackings = np.empty((points.shape[0], *points.shape), dtype=points.dtype)

    stackings[0] = points

    corresponding = np.arange(points.shape[1]) # vectorized equivalent of ":"

    for n in range(points.shape[0] - 1):
        stackings[n + 1] = stackings[n]
        stackings[n + 1, order[n], corresponding] += period

    # return most localized stacking:

    localized = np.argmin(np.std(stackings, axis=1), axis=0)

    return stackings[localized, :, corresponding].T.reshape(shape)

    # re "corresponding" and ".T" see NumPy's "Advanced indexing" and "NEP 21"

def linear_interpolation(data, angle=60, axes=(0, 1), period=None, polar=False):
    """Perform linear interpolation in one or two dimensions.

    The edges are interpolated using periodic boundary conditions.

    Parameters
    ----------
    data : ndarray
        Data on uniform 1D or 2D (triangular or rectangular) lattice.
    angle : number
        Angle between lattice vectors in degrees.
    axes : int or 2-tuple of int
        Axes of `data` along which to interpolate (lattice vectors).
    period : number
        If the values of `data` are defined on a periodic axis (i.e., only with
        respect to the modulo operation), the period of this axis. This is used
        in combination with `stack` to always interpolate across the shortest
        distance of two neighboring points.
    polar : bool
        Interpolate complex values linearly in polar coordinates? This is
        helpful if neighboring data points have arbitrary complex phases.

    Returns
    -------
    function
        Interpolant for `data`. ``linear_interpolation(data)(i, j)`` yields the
        same value as ``data[i, j]``. Thus the data array is "generalized" with
        respect to fractional indices.

    See Also
    --------
    stack : Condense point cloud on periodic axis.
    resize : Compress or stretch data via linear interpolation.
    Fourier_interpolation : Alternative interpolation routine.
    """
    if polar and np.iscomplexobj(data):
        r = linear_interpolation(abs(data), angle, axes, period=period)
        p = linear_interpolation(np.angle(data), angle, axes, period=2 * np.pi)

        def interpolant(*args, **kwargs):
            return r(*args, **kwargs) * np.exp(1j * p(*args, **kwargs))

        return interpolant

    if not hasattr(axes, '__len__'):
        axes = (axes,)

    # move lattice axes to the front:

    order = tuple(axes) + tuple(n for n in range(data.ndim) if n not in axes)

    data = np.transpose(data, axes=order)

    # interpret "fractional indices":

    def split(n, N):
        n0, dn = divmod(n, 1)
        n0 = int(n0) % N

        return n0, dn

    # define interpolation routines for different lattices:

    N = data.shape[0]

    if len(axes) > 1:
        M = data.shape[1]

    if len(axes) == 1:
        #  ______
        # A  a1  B
        #
        def interpolant(n):
            n0, dn = split(n, N)

            A = data[n0]
            B = data[(n0 + 1) % N]

            if period:
                A, B = stack(A, B, period=period)

            return (1 - dn) * A + dn * B

    elif angle == 60:
        #
        #     B______C'
        #     /\    /
        # a2 /  \  /
        #   /____\/
        #  C  a1  A
        #
        def interpolant(n, m):
            n0, dn = split(n, N)
            m0, dm = split(m, M)

            A = data[(n0 + 1) % N, m0]
            B = data[n0, (m0 + 1) % M]

            prime = dn + dm > 1 # use C' rather than C

            if prime:
                C = data[(n0 + 1) % N, (m0 + 1) % M]
            else:
                C = data[n0, m0]

            if period:
                A, B, C = stack(A, B, C, period=period)

            if prime:
                return (1 - dm) * A + (1 - dn) * B + (dn + dm - 1) * C
            else:
                return dn * A + dm * B + (1 - dn - dm) * C

    elif angle == 90:
        #
        #   D ____ C
        #    |    |
        # a2 |    |
        #    |____|
        #   A  a1  B
        #
        def interpolant(n, m):
            n0, dn = split(n, N)
            m0, dm = split(m, M)

            A = data[n0, m0]
            B = data[(n0 + 1) % N, m0]
            C = data[(n0 + 1) % N, (m0 + 1) % M]
            D = data[n0, (m0 + 1) % M]

            if period:
                A, B, C, D = stack(A, B, C, D, period=period)

            return ((1 - dn) * (1 - dm) * A + dn * (1 - dm) * B
                + dn * dm * C + (1 - dn) * dm * D)

    elif angle == 120:
        #
        #  C______B
        #   \    /\
        # a2 \  /  \
        #     \/____\
        #     A  a1  C'
        #
        def interpolant(n, m):
            n0, dn = split(n, N)
            m0, dm = split(m, M)

            A = data[n0, m0]
            B = data[(n0 + 1) % N, (m0 + 1) % M]

            prime = dn > dm # use C' rather than C

            if prime:
                C = data[(n0 + 1) % N, m0]
            else:
                C = data[n0, (m0 + 1) % M]

            if period:
                A, B, C = stack(A, B, C, period=period)

            if prime:
                return (1 - dn) * A + dm * B + (dn - dm) * C
            else:
                return (1 - dm) * A + dn * B + (dm - dn) * C

    # make interpolant function applicable to arrays and return:

    return np.vectorize(interpolant)

def resize(data, shape, angle=60, axes=(0, 1), period=None, polar=False,
        periodic=True):
    """Resize array via linear interpolation along one or two axes.

    Parameters
    ----------
    shape : int or 2-tuple of int
        New lattice shape.
    shape, angle, axes, period, polar
        Parameters for :func:`linear_interpolation`.
    periodic : bool, default True
        Interpolate between last and first data point of each axis?

    Returns
    -------
    ndarray
        Resized data array.

    See Also
    --------
    linear_interpolation
    """
    if not hasattr(shape, '__len__'):
        shape = (shape,)

    if not hasattr(axes, '__len__'):
        axes = (axes,)

    # move lattice axes to the front:

    order = tuple(axes) + tuple(n for n in range(data.ndim) if n not in axes)

    data = np.transpose(data, axes=order)

    # set up interpolation function:

    interpolant = linear_interpolation(data, angle, axes=range(len(shape)),
        period=period, polar=polar)

    # apply interpolation function at new lattice points in parallel:

    size = np.prod(shape)
    sizes, bounds = elphmod.MPI.distribute(size, bounds=True)

    my_new_data = np.empty((sizes[comm.rank], *data.shape[len(shape):]),
        dtype=data.dtype)

    if periodic:
        scale = np.divide(data.shape[:len(shape)], shape)
    else:
        scale = np.subtract(data.shape[:len(shape)], 1) / np.subtract(shape, 1)

    for n, m in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        if len(shape) == 2:
            m = m // shape[1], m % shape[1]

        my_new_data[n] = interpolant(*np.multiply(m, scale))

    new_data = np.empty((*shape, *data.shape[len(shape):]),
        dtype=data.dtype)

    comm.Allgatherv(my_new_data,
        (new_data, sizes * np.prod(data.shape[len(shape):], dtype=int)))

    # restore original order of axes and return:

    new_data = np.transpose(new_data, axes=np.argsort(order))

    return new_data

def squared_distance(k1, k2, angle=60):
    """Calculate squared distance of lattice point from origin.

    If the coordinates are given as integers, the result is numerically exact.
    For 60 or 120 deg. (triangular lattice) this yields the Loeschian numbers.
    Non-equivalent lattice sites may have the same distance from the origin!
    (E.g., there are non-equivalent 20th neighbors in a triangular lattice.)

    Parameters
    ----------
    k1, k2 : int
        Point in lattice coordinates (crystal coordinates, mesh-indices, ...).
    angle : number
        Angle between lattice axes.

    Returns
    -------
    number
        Squared distance of point from origin.
    """
    sgn = {60: 1, 90: 0, 120: -1}[angle]

    return k1 * k1 + k2 * k2 + sgn * k1 * k2

def to_Voronoi(k1, k2, nk, angle=60, dk1=0, dk2=0, epsilon=0.0):
    """Map any lattice point to the Voronoi cell* around the origin.

    (*) Wigner-Seitz cell/Brillouin zone for Bravais/reciprocal lattice.

    Parameters
    ----------
    k1, k2 : int
        Mesh-point indices.
    nk : int
        Number of points per dimension.
    angle : number
        Angle between lattice vectors.
    dk1, dk2 : number
        Shift of Voronoi cell.
    epsilon : float
        Maximum absolute difference of "equal" floats.

    Returns
    -------
    set
        All equivalent point within or on the edge of the Voronoi cell.
    """
    k1 %= nk
    k2 %= nk

    images = [(k1, k2), (k1 - nk, k2), (k1, k2 - nk), (k1 - nk, k2 - nk)]

    if nk < 3:
        images.extend([(k1 + nk, k2), (k1, k2 + nk), (k1 + nk, k2 + nk)])

    distances = [squared_distance(k1 - dk1, k2 - dk2, angle=angle)
        for k1, k2 in images]
    minimum = min(distances)
    images = {image for image, distance in zip(images, distances)
        if abs(distance - minimum) <= epsilon}

    return images

def wigner_2d(nk, angle=120, dk1=0.0, dk2=0.0, epsilon=0.0):
    """Find lattice points in Wigner-Seitz cell (including boundary).

    Parameters
    ----------
    nk : int
        Number of points per dimension.
    angle : number
        Angle between lattice vectors.
    dk1, dk2 : float
        Shift of Wigner-Seitz cell.
    epsilon : float
        Maximum absolute difference of "equal" floats.

    Returns
    -------
    list of tuple of int
        Mesh-point indices.
    list of int
        Degeneracies.
    list of float
        Lattice-vector lengths.
    """
    points = []

    for k1 in range(nk):
        for k2 in range(nk):
            images = to_Voronoi(k1, k2, nk, angle, dk1, dk2, epsilon)

            points.extend([(point, len(images)) for point in images])

    points = sorted(points)

    irvec, ndegen = zip(*points)

    wslen = [np.sqrt(squared_distance(k1, k2, angle)) for k1, k2 in irvec]

    return irvec, ndegen, wslen

def read_wigner_file(name, old_ws=False, nat=None):
    """Read binary file with Wigner-Seitz data as used by EPW.

    Parameters
    ----------
    name : str
        Name of file with Wigner-Seitz data.
    old_ws : bool
        Use previous definition of Wigner-Seitz cells? This is required if
        `patches/qe-6.3-backports.patch` has been used.
    nat : int
        Number of atoms per unit cell.

    See Also
    --------
    elphmod.elph.Model
    """
    if comm.rank == 0:
        try:
            data = open(name, 'rb')
        except FileNotFoundError:
            if name.endswith('.dat'):
                other = name[:-4] + '.fmt'
            elif name.endswith('.fmt'):
                other = name[:-4] + '.dat'
            else:
                raise

            print('Warning: "%s" not found, trying "%s"!' % (name, other))
            name = other

            data = open(name, 'rb')

        binary = b'\x00' in data.read()
        data.close()

        if binary:
            with open(name, 'rb') as data:
                integer = np.int32
                double = np.float64

                if old_ws:
                    dims = 1
                    dims2 = nat
                else:
                    dims, = np.fromfile(data, integer, 1)
                    dims2, = np.fromfile(data, integer, 1)

                nrr_k, = np.fromfile(data, integer, 1)
                irvec_k = np.fromfile(data, integer, nrr_k * 3)
                irvec_k = irvec_k.reshape((nrr_k, 3))
                ndegen_k = np.fromfile(data, integer, dims ** 2 * nrr_k)
                ndegen_k = ndegen_k.reshape((dims, dims, nrr_k))

                if old_ws:
                    wslen_k = np.fromfile(data, double, nrr_k)
                    nrr_q, = np.fromfile(data, integer, 1)
                    irvec_q = np.fromfile(data, integer, nrr_q * 3)
                    irvec_q = irvec_q.reshape((nrr_q, 3))
                    ndegen_q = np.fromfile(data, integer, dims2 * dims2 * nrr_q)
                    ndegen_q = ndegen_q.reshape((dims2, dims2, nrr_q))
                    wslen_q = np.fromfile(data, double, nrr_q)

                nrr_g, = np.fromfile(data, integer, 1)
                irvec_g = np.fromfile(data, integer, nrr_g * 3)
                irvec_g = irvec_g.reshape((nrr_g, 3))

                if old_ws:
                    ndegen_g = np.fromfile(data, integer, dims2 * nrr_g)
                    wslen_g = np.fromfile(data, double, nrr_g)
                else:
                    ndegen_g = np.fromfile(data, integer)

                try:
                    ndegen_g = ndegen_g.reshape((dims, dims, dims2, nrr_g))
                except ValueError: # since QE 6.8
                    ndegen_g = ndegen_g.reshape((dims2, nrr_g, 1, dims))
                    ndegen_g = ndegen_g.transpose((2, 3, 0, 1))
        else:
            with open(name, 'r') as lines:
                def integers(n=None):
                    return list(map(int, next(lines).split()[:n]))

                nrr_k, nrr_q, nrr_g, dims, dims2 = integers()

                irvec_k = np.empty((nrr_k, 3), dtype=np.int32)
                irvec_q = np.empty((nrr_q, 3), dtype=np.int32)
                irvec_g = np.empty((nrr_g, 3), dtype=np.int32)

                ndegen_k = np.empty((dims, dims, nrr_k), dtype=np.int32)
                ndegen_q = np.empty((dims2, dims2, nrr_q), dtype=np.int32)
                ndegen_g = np.empty((1, dims, dims2, nrr_g), dtype=np.int32)

                for ir in range(nrr_k):
                    irvec_k[ir] = integers(3)

                    for iw in range(dims):
                        ndegen_k[:, iw, ir] = integers()

                for ir in range(nrr_q):
                    irvec_q[ir] = integers(3)

                    for na in range(dims2):
                        ndegen_q[:, na, ir] = integers()

                for ir in range(nrr_g):
                    irvec_g[ir] = integers(3)

                    for iw in range(dims):
                        ndegen_g[0, iw, :, ir] = integers()

        data = irvec_k, ndegen_k, irvec_g, ndegen_g
    else:
        data = None

    data = comm.bcast(data)

    return data

def wigner(nr1, nr2, nr3, at, tau, tau2=None, eps=1e-7, sgn=+1, nsc=2):
    """Determine Wigner-Seitz lattice vectors with degenercies and lengths.

    Parameters
    ----------
    nr1, nr2, nr3 : ndarray
        Dimensions of positive lattice-vector mesh.
    at : ndarray
        Bravais lattice vectors.
    tau : ndarray
        Positions of basis orbitals or atoms in original cell.
    tau2 : ndarray, optional
        Positions of basis orbitals or atoms in shifted cell. Defaults to `tau`.
    eps : float
        Tolerance for orbital or atomic distances to be considered equal.
    sgn : int
        Do the lattice vectors shift the first (``-1``) or second (``+1``)
        orbital/atom?
    nsc : int
        Number of supercells per dimension and direction where Wigner-Seitz
        lattice vectors are searched for.

    Returns
    -------
    irvec : ndarray
        Wigner-Seitz lattice vectors.
    ndegen : ndarray
        Corresponding degeneracies.
    wslen : ndarray
        Corresponding lengths.
    """
    if tau2 is None:
        tau2 = tau

    supercells = range(-nsc, nsc) # intentionally asymmetric (mesh is positive)

    data = dict()

    N = 0 # counter for parallelization

    for m1 in range(nr1):
        for m2 in range(nr2):
            for m3 in range(nr3):
                N += 1

                if N % comm.size != comm.rank:
                    continue

                # determine equivalent unit cells within considered supercells:

                copies = sgn * np.array([
                    [
                        M1 * nr1 + m1,
                        M2 * nr2 + m2,
                        M3 * nr3 + m3,
                    ]
                    for M1 in supercells
                    for M2 in supercells
                    for M3 in supercells
                ])

                # calculate corresponding translation vectors:

                shifts = copies.dot(at)

                for i in range(len(tau)):
                    for j in range(len(tau2)):
                        # find equivalent bond(s) within Wigner-Seitz cell:

                        bonds = [r + tau2[j] - tau[i] for r in shifts]
                        lengths = [np.sqrt(np.dot(r, r)) for r in bonds]
                        length = min(lengths)

                        selected = copies[np.where(abs(lengths - length) < eps)]

                        # save mapped lattice vectors and degeneracy and length:

                        for R in selected:
                            R = tuple(R)

                            if R not in data:
                                data[R] = [
                                    np.zeros((len(tau), len(tau2)), dtype=int),
                                    np.zeros((len(tau), len(tau2)))]

                            data[R][0][i, j] = len(selected)
                            data[R][1][i, j] = length

    # convert dictionary into arrays:

    my_count = len(data)
    my_irvec = np.array(list(data.keys()), dtype=int)
    my_ndegen = np.empty((my_count, len(tau), len(tau2)), dtype=int)
    my_wslen = np.empty((my_count, len(tau), len(tau2)))

    for i, (d, l) in enumerate(data.values()):
        my_ndegen[i] = d
        my_wslen[i] = l

    # gather data of all processes:

    my_counts = np.array(comm.allgather(my_count))
    count = my_counts.sum()

    irvec = np.empty((count, 3), dtype=int)
    ndegen = np.empty((count, len(tau), len(tau2)), dtype=int)
    wslen = np.empty((count, len(tau), len(tau2)))

    comm.Allgatherv(my_irvec, (irvec, my_counts * 3))
    comm.Allgatherv(my_ndegen, (ndegen, my_counts * len(tau) * len(tau2)))
    comm.Allgatherv(my_wslen, (wslen, my_counts * len(tau) * len(tau2)))

    # (see cdef _p_message message_vector in mpi4py/src/mpi4py/MPI/msgbuffer.pxi
    # for possible formats of second argument 'recvbuf')

    return irvec, ndegen, wslen

def short_range_model(data, at, tau, sgn=+1, divide_ndegen=True):
    """Map hoppings or force constants onto Wigner-Seitz cell.

    Parameters
    ----------
    data : ndarray
        Hoppings or force constants on (positive) Fourier-transform mesh. The
        first two dimensions correspond to the orbitals or atoms, the following
        three to the mesh axes, and the last two (optional, relevant for force
        constants only) to Cartesian directions.
    at : ndarray
        Bravais lattice vectors.
    tau : ndarray
        Positions of basis orbitals or atoms.
    sgn : int
        Do the lattice vectors shift the first (``-1``) or second (``+1``)
        orbital/atom?
    divide_ndegen : bool
        Divide hoppings for force constants by lattice-vector degeneracy?

    Returns
    -------
    irvec : ndarray
        Wigner-Seitz lattice vectors.
    const : ndarray
        Corresponding hoppings or force constants.
    wslen : ndarray
        Corresponding lengths.
    """
    while data.ndim < 7:
        data = data[..., np.newaxis]

    nbasis, nbasis, nr1, nr2, nr3, ncart, ncart = data.shape

    irvec, ndegen, wslen = wigner(nr1, nr2, nr3, at, tau, sgn=sgn)

    sizes, bounds = elphmod.MPI.distribute(len(irvec), bounds=True)

    my_const = np.zeros((sizes[comm.rank], nbasis * ncart, nbasis * ncart),
        dtype=data.dtype)

    for my_n, n in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        m1 = sgn * irvec[n, 0] % nr1
        m2 = sgn * irvec[n, 1] % nr2
        m3 = sgn * irvec[n, 2] % nr3

        for i in range(nbasis):
            for j in range(nbasis):
                tmp = my_const[my_n,
                    i * ncart:(i + 1) * ncart,
                    j * ncart:(j + 1) * ncart]

                tmp[:, :] = data[i, j, m1, m2, m3]

                if divide_ndegen:
                    if ndegen[n, i, j]:
                        tmp /= ndegen[n, i, j]
                    else:
                        tmp[:, :] = 0.0

    const = np.zeros((len(irvec), nbasis * ncart, nbasis * ncart),
        dtype=data.dtype)

    comm.Allgatherv(my_const, (const, comm.allgather(my_const.size)))

    return irvec, const, wslen

def Fourier_interpolation(data, angle=60, sign=-1, function=True):
    """Perform Fourier interpolation on triangular or rectangular lattice.

    Parameters
    ----------
    data : ndarray
        Data on uniform triangular or rectangular lattice.
    angle : number
        Angle between lattice vectors in degrees.
    sign : number
        Sign in exponential function in first Fourier transform.
    function : bool
        Return interpolation function or parameter dictionary?

    Returns
    -------
    function
        Interpolant for `data`. ``Fourier_interpolation(data)(i, j)`` yields
        the same value as ``data[i, j]``. Thus the data array is "generalized"
        with respect to fractional indices.

    See Also
    --------
    linear_interpolation : Alternative interpolation routine.
    """
    N, N = data.shape[:2]

    # do first Fourier transform to obtain coefficients:

    i = np.arange(N)

    transform = np.exp(sign * 2j * np.pi / N * np.outer(i, i)) / N

    data = np.einsum('ni,ij...,jm->nm...', transform, data, transform)

    # construct smooth inverse transform (formally tight-binding model):

    values = np.empty((N * N * 4, *data.shape[2:]), dtype=complex)
    points = np.empty((N * N * 4, 2), dtype=int)
    counts = np.empty((N * N * 4), dtype=int)

    count = 0
    for n in range(N):
        for m in range(N):
            if np.all(abs(data[n, m]) < 1e-6):
                continue

            images = to_Voronoi(n, m, N, angle=180 - angle)

            # angle transform: from real to reciprocal lattice or vice versa

            for point in images:
                values[count] = data[n, m]
                points[count] = point
                counts[count] = len(images)
                count += 1

    values = values[:count]
    points = points[:count]
    counts = counts[:count]

    # fix weights of interpolation coefficients:

    for i in range(count):
        values[i] /= counts[i]

    # define interpolation function and generalize is with respect to arrays:

    idphi = -sign * 2j * np.pi / N

    def interpolant(*point):
        return np.dot(np.exp(idphi * np.dot(points, point)), values).real

    if function:
        return np.vectorize(interpolant)

    # return either interpolation function or parameter dictionary:

    return dict((tuple(point), value) for point, value in zip(points, values))

def path(points, N=30, recvec=None, qe=False, moveG=0, **kwargs):
    r"""Generate arbitrary path through Brillouin zone.

    Parameters
    ----------
    points : ndarray
        List of high-symmetry points in crystal coordinates. Some well-known
        labels such as ``G`` (|Ggr|), ``M``, or ``K`` may also be used. Mostly,
        the definitions follow https://lampx.tugraz.at/~hadley/ss1/bzones/.
    N : float
        Number of points per :math:`2 \pi / a`.
    recvec : ndarray, optional
        List of reciprocal lattice vectors.
    qe : bool, default False
        Also return path in QE input format?
    moveG : float, default 0
        Move Gamma point to the closest nonzero point multiplied by this value.
        This is useful, e.g., to plot phonon dispersions with TO-LO splitting.
    **kwargs
        Arguments passed to :func:`primitives`, e.g., parameters from
        'func'`read_pwi`, particularly the Bravais-lattice index `ibrav`.

    Returns
    -------
    ndarray
        Points in crystal coordinates with period :math:`2 \pi`.
    ndarray
        Cumulative path distance.
    list
        Indices of corner/high-symmetry points.
    dict, optional
        Path in format suitable for :func:`write_pwi`.

    See Also
    --------
    primitives
    reciprocals
    """
    labels = {
        1: { # cubic (sc)
            'M': [0.0, 0.5, 0.5],
            'R': [0.5, 0.5, 0.5],
            'X': [0.0, 0.0, 0.5],
        },
        2: { # cubic (fcc)
            'X': [0.0, 0.5, 0.5],
            'L': [0.5, 0.5, 0.5],
            'W': [0.25, 0.75, 0.5],
            'U': [0.25, 0.625, 0.625],
            'K': [0.375, 0.75, 0.375],
        },
        3: { # cubic (bcc) (*)
            'H': [0.5, 0.5, 0.5],
            'P': [0.75, 0.25, -0.25],
            'N': [0.5, 0.5, 0.0],
        },
        -3: { # cubic (bcc, more symmetric axis)
            'H': [0.5, 0.5, -0.5],
            'P': [0.25, 0.25, 0.25],
            'N': [0.5, 0.0, 0.0],
        },
        4: { # hexagonal & trigonal
            'A': [0.0, 0.0, 0.5],
            'M': [0.0, 0.5, 0.0],
            'L': [0.0, 0.5, 0.5],
            'K': [1 / 3, 1 / 3, 0.0],
            'H': [1 / 3, 1 / 3, 0.5],
        },
        6: { # tetragonal (st)
            'X': [0.5, 0.0, 0.0],
            'M': [0.5, 0.5, 0.0],
            'Z': [0.0, 0.0, 0.5],
            'R': [0.5, 0.0, 0.5],
            'A': [0.5, 0.5, 0.5],
        },
        7: { # tetragonal (bct) (generated from 6)
            'X': [0.25, 0.25, -0.25],
            'M': [0.00, 0.50, -0.50],
            'Z': [0.25, 0.25, 0.25],
            'R': [0.50, 0.50, 0.00],
            'A': [0.25, 0.75, -0.25],
        },
        8: { # orthorhombic
            'X': [0.5, 0.0, 0.0],
            'Y': [0.0, 0.5, 0.0],
            'Z': [0.0, 0.0, 0.5],
            'S': [0.5, 0.5, 0.0],
            'T': [0.0, 0.5, 0.5],
            'U': [0.5, 0.0, 0.5],
            'R': [0.5, 0.5, 0.5],
        },
        9: { # orthorhombic (bco) (*)
            'Y': [0.5, -0.5, 0.0],
            'y': [0.5, 0.5, 0.0],
            'Z': [0.0, 0.0, 0.5],
            'T': [0.5, -0.5, 0.5],
            't': [0.5, 0.5, 0.5],
            'S': [0.5, 0.0, 0.0],
            'R': [0.5, 0.0, 0.5],
        },
        -9: { # orthorhombic (bco, alternate description)
            'Y': [0.5, 0.5, 0.0],
            'y': [-0.5, 0.5, 0.0],
            'Z': [0.0, 0.0, 0.5],
            'T': [0.5, 0.5, 0.5],
            't': [-0.5, 0.5, 0.5],
            'S': [0.0, 0.5, 0.0],
            'R': [0.0, 0.5, 0.5],
        },
        12: { # monoclinic (unique axis c) (*)
            'B': [-0.5, 0.0, 0.0],
            'Y': [0.0, 0.5, 0.0],
            'Z': [0.0, 0.0, 0.5],
            'D': [-0.5, 0.5, 0.0],
            'C': [0.0, 0.5, 0.5],
            'A': [-0.5, 0.0, 0.5],
            'E': [-0.5, 0.5, 0.5],
        },
        -12: { # monoclinic (unique axis b)
            'B': [-0.5, 0.0, 0.0],
            'Y': [0.0, 0.5, 0.0],
            'Z': [0.0, 0.0, 0.5],
            'D': [-0.5, 0.5, 0.0],
            'C': [0.0, 0.5, 0.5],
            'A': [-0.5, 0.0, 0.5],
            'E': [-0.5, 0.5, 0.5],
        },
    }.get(kwargs.get('ibrav', 8), {})

    # (*) generated from points for Bravais lattice with index of opposite sign
    # through X'[k] = X[i] b[i, j] a'[k, j], where b are the reciprocals of a.

    labels['G'] = [0.0, 0.0, 0.0]

    points = np.array([labels[point] if isinstance(point, str) else point
        for point in points])

    if recvec is None:
        a = primitives(**kwargs)
        b = reciprocals(*a)
    else:
        b = recvec

    points_cart = np.einsum('kc,cx->kx', points, b)

    k = []
    x = []
    corners = [0]

    x0 = 0.0

    for i in range(len(points) - 1):
        dx = np.linalg.norm(points_cart[i + 1] - points_cart[i])
        n = max(1, int(round(N * dx)))

        x1 = x0 + dx

        j0 = 0 if i == 0 or moveG and np.all(points[i] == 0) else 1

        for j in range(j0, n + 1):
            k.append((j * points[i + 1] + (n - j) * points[i]) / n)
            x.append((j * x1 + (n - j) * x0) / n)

        x0 = x1

        corners.append(len(k) - 1)

    if moveG:
        for i in range(len(points)):
            if np.all(points[i] == 0):
                if i == 0:
                    k[corners[i]] += moveG * k[corners[i] + 1]
                else:
                    k[corners[i]] += moveG * k[corners[i] - 1]

                    if i != len(corners) - 1:
                        k[corners[i] + 1] += moveG * k[corners[i] + 2]

    if qe:
        kwargs['ktyp'] = 'crystal_b'
        kwargs['nks'] = len(corners)
        kwargs['k_points'] = np.empty((len(corners), 4))

        for i in range(len(corners)):
            kwargs['k_points'][i, :3] = k[corners[i]]
            try:
                kwargs['k_points'][i, 3] = corners[i + 1] - corners[i]
            except IndexError:
                kwargs['k_points'][i, 3] = 0

        return 2 * np.pi * np.array(k), np.array(x), corners, kwargs
    else:
        return 2 * np.pi * np.array(k), np.array(x), corners

def GMKG(N=30, corner_indices=False, mesh=False, angle=60, straight=True,
        lift_degen=True):
    r"""Generate path |Ggr|-M-K-|Ggr| through Brillouin zone.

    (This function should be replaced by a more general one to produce arbitrary
    paths through both triangular and rectangular Brillouin zones, where special
    points can be defined using labels such as ``'G'``, ``'M'``, or ``'K'``.)

    Parameters
    ----------
    N : int
        Number of mesh points per dimension if `mesh` is ``True`` and `N` is a
        multiple of 6. Otherwise the number of points per :math:`2 \pi / a`.
    corner_indices : bool
        Return indices of corner/high-symmetry points?
    mesh : bool
        Return points of uniform mesh that exactly lie on the path? If ``True``,
        `N` must be a multiple of 6.
    angle : number
        Angle between reciprocal basis lattice vectors.
    straight : bool
        Cross K in a straight line? In this case, the path does not enclose the
        irreducible wedge.
    lift_degen : bool
        Lift degeneracy at K by infinitesimal shift toward |Ggr|?

    Returns
    -------
    ndarray
        Points in crystal coordinates with period :math:`2 \pi`.
    ndarray
        Cumulative path distance.
    list, optional
        Indices of corner/high-symmetry points.
    """
    G = 2 * np.pi * np.array([0.0, 0.0])
    M = 2 * np.pi * np.array([0.0, 1.0]) / 2

    if angle == 60:
        K = 2 * np.pi * np.array([ 1.0, 1.0]) / 3
        k = 2 * np.pi * np.array([-2.0, 1.0]) / 3

    else:
        K = 2 * np.pi * np.array([ 1.0,  2.0]) / 3
        k = 2 * np.pi * np.array([-2.0, -1.0]) / 3

    if not straight:
        k = K

    if lift_degen:
        shrink = 1.0 - 1e-10

        k *= shrink
        K *= shrink

    L1 = 1 / np.sqrt(3)
    L2 = 1 / 3
    L3 = 2 / 3

    if mesh and not N % 6: # use only points of N x N mesh
        N1 = N // 2
        N2 = N // 6
        N3 = N // 3
    else:
        N1 = int(round(N * L1))
        N2 = int(round(N * L2))
        N3 = int(round(N * L3))

    N3 += 1

    def line(k1, k2, N=100, endpoint=True):
        q1 = np.linspace(k1[0], k2[0], N, endpoint)
        q2 = np.linspace(k1[1], k2[1], N, endpoint)

        return list(zip(q1, q2))

    path = line(G, M, N1, False) \
         + line(M, K, N2, False) \
         + line(k, G, N3, True)

    x = np.empty(N1 + N2 + N3)

    x[0:N1] = np.linspace(0, L1, N1, False)
    x[N1:N1 + N2] = np.linspace(L1, L1 + L2, N2, False)
    x[N2 + N1:N1 + N2 + N3] = np.linspace(L2 + L1, L1 + L2 + L3, N3, True)

    if corner_indices:
        return np.array(path), x, [0, N1, N1 + N2, N1 + N2 + N3 - 1]
    else:
        return np.array(path), x

def BZ(angle=120, angle0=0):
    """Draw Brillouin zone outline."""

    a1, a2 = translations(angle=angle, angle0=angle0)
    b1, b2 = reciprocals(a1, a2)

    if angle == 60:
        K = [(1, 2), (-1, 1), (-2, -1), (-1, -2), (1, -1), (2, 1), (1, 2)]
        outline = np.array([k1 * b1 + k2 * b2 for k1, k2 in K]) / 3

    elif angle == 90:
        M = [(1, 1), (-1, 1), (-1, -1), (1, -1), (1, 1)]
        outline = np.array([k1 * b1 + k2 * b2 for k1, k2 in M])

    elif angle == 120:
        K = [(1, 1), (-1, 2), (-2, 1), (-1, -1), (1, -2), (2, -1), (1, 1)]
        outline = np.array([k1 * b1 + k2 * b2 for k1, k2 in K]) / 3

    return outline

def read_pwi(pwi):
    """Read input data and crystal structure from PWscf input file.

    Parameters
    ----------
    pwi : str
        Filename.

    Returns
    -------
    dict
        Input data and crystal structure.
    """
    struct = elphmod.misc.read_input_data(pwi, broadcast=False)

    if comm.rank == 0:
        for key in ['celldm']:
            if key in struct and not isinstance(struct[key], list):
                struct[key] = [struct[key]]

        with open(pwi) as lines:
            for line in lines:
                words = [word for word in line.split() if word]

                if not words:
                    continue

                key = words[0].lower()

                if key == 'atomic_species':
                    struct['at_species'] = []
                    struct['mass'] = []
                    struct['pp'] = []

                    for n in range(struct['ntyp']):
                        words = next(lines).split()

                        struct['at_species'].append(words[0])
                        struct['mass'].append(float(words[1]))
                        struct['pp'].append(words[2])

                elif key == 'atomic_positions':
                    if len(words) == 1:
                        struct['coords'] = 'alat'
                    else:
                        struct['coords'] = ''.join(c for c in words[1].lower()
                            if c not in '{ }')

                    print("Read crystal structure in units '%s'"
                        % struct['coords'])

                    struct['at'] = []

                    if 'crystal_sg' in struct['coords']:
                        struct['r'] = []
                    else:
                        struct['r'] = np.empty((struct['nat'], 3))

                    for n in range(struct['nat']):
                        words = next(lines).split()

                        struct['at'].append(words[0])

                        if 'crystal_sg' in struct['coords']:
                            struct['r'].append(words[1])
                        else:
                            for x in range(3):
                                struct['r'][n, x] = float(words[1 + x])

                elif key == 'k_points':
                    struct['ktyp'] = words[1].lower()

                    if 'automatic' in struct['ktyp']:
                        struct[key] = list(map(int, next(lines).split()[:6]))
                    elif 'gamma' not in struct['ktyp']:
                        struct['nks'] = int(next(lines))
                        struct[key] = np.empty((struct['nks'], 4))

                        for n in range(struct['nks']):
                            struct[key][n] = list(map(float,
                                next(lines).split()[:4]))

                elif key == 'cell_parameters':
                    struct['r_cell'] = np.empty((3, 3))

                    if len(words) == 1:
                        struct['cell_units'] = 'bohr'
                    else:
                        struct['cell_units'] = words[1]

                    print("Read cell parameters in units '%s'"
                        % struct['cell_units'])

                    for n in range(3):
                        words = next(lines).split()

                        for x in range(3):
                            struct['r_cell'][n, x] = float(words[x])

    struct = comm.bcast(struct)

    return struct

def write_pwi(pwi, struct):
    """Write crystal structure to PWscf input file.

    Parameters
    ----------
    pwi : str
        Filename.
    struct : dict
        Crystal structure.
    """
    if comm.rank != 0:
        return

    with open(pwi, 'w') as data:
        data.write('&CONTROL\n')

        for key in ['title', 'prefix', 'outdir', 'pseudo_dir', 'calculation',
                'tprnfor', 'tstress', 'nstep', 'forc_conv_thr']:
            if key in struct:
                data.write('%s = %r\n' % (key, struct[key]))

        data.write('/\n')

        data.write('&SYSTEM\n')

        if 'celldm' in struct:
            for i, celldm in enumerate(struct['celldm'], 1):
                if celldm:
                    data.write('celldm(%d) = %r\n' % (i, celldm))

        for key in ['ibrav', 'ntyp', 'nat', 'a', 'b', 'c', 'cosbc', 'cosac',
                'cosab', 'ecutwfc', 'ecutrho', 'nbnd', 'occupations',
                'smearing', 'degauss', 'nosym', 'noinv', 'tot_charge',
                'assume_isolated', 'nspin', 'noncolin', 'lspinorb',
                'tot_magnetization', 'starting_magnetization', 'space_group',
                'uniqueb', 'origin_choice', 'rhombohedral']:
            if key in struct:
                data.write('%s = %r\n' % (key, struct[key]))

        data.write('/\n')

        data.write('&ELECTRONS\n')

        for key in ['electron_maxstep', 'conv_thr', 'diagonalization',
                'diago_full_acc', 'mixing_beta', 'startingpot', 'startingwfc',
                'scf_must_converge']:
            if key in struct:
                data.write('%s = %r\n' % (key, struct[key]))

        data.write('/\n')

        if struct.get('calculation') in {'relax', 'md', 'vc-relax', 'vc-md'}:
            data.write('&IONS\n')

            for key in ['ion_dynamics', 'upscale']:
                if key in struct:
                    data.write('%s = %r\n' % (key, struct[key]))

            data.write('/\n')

            if struct['calculation'] in {'vc-relax', 'vc-md'}:
                data.write('&CELL\n')

                for key in ['cell_dynamics', 'cell_dofree', 'press',
                        'press_conv_thr']:
                    if key in struct:
                        data.write('%s = %r\n' % (key, struct[key]))

                data.write('/\n')

        data.write('ATOMIC_SPECIES\n')

        for i in range(struct['ntyp']):
            data.write('%s %.12g %s\n'
                % (struct['at_species'][i], struct['mass'][i], struct['pp'][i]))

        data.write('\n')

        data.write('ATOMIC_POSITIONS %s\n' % struct['coords'])

        if 'crystal_sg' in struct['coords'].lower():
            for Xr in zip(struct['at'], struct['r']):
                data.write('%2s %s\n' % Xr)
        else:
            for X, (r1, r2, r3) in zip(struct['at'], struct['r']):
                data.write('%2s %12.9f %12.9f %12.9f\n' % (X, r1, r2, r3))

        if 'ktyp' in struct:
            data.write('\n')

            data.write('K_POINTS %s\n' % struct['ktyp'])

            if 'automatic' in struct['ktyp']:
                data.write('%d %d %d %d %d %d\n' % tuple(struct['k_points']))
            elif 'gamma' not in struct['ktyp']:
                data.write('%d\n' % struct['nks'])

                for (kx, ky, kz, wk) in struct['k_points']:
                    data.write('%12.9f %12.9f %12.9f ' % (kx, ky, kz))

                    if wk == int(wk):
                        data.write('%d\n' % wk)
                    else:
                        data.write('%12.9f\n' % wk)

        if 'r_cell' in struct:
            data.write('\n')

            data.write('CELL_PARAMETERS %s\n' % struct['cell_units'])

            for r in struct['r_cell']:
                data.write('%12.9f %12.9f %12.9f\n' % tuple(r))

def read_win(win):
    """Read input data from .win file (Wannier90).

    Parameters
    ----------
    win : str
        Filename.

    Returns
    -------
    dict
        Crystal structure.
    """
    if comm.rank == 0:
        struct = dict()

        with open(win) as lines:
            for line in lines:
                words = [word
                    for column in line.split()
                    for word in column.split('=') if word]

                if not words:
                    continue

                key = words[0].lower()

                if key == 'num_bands':
                    struct[key] = int(words[1])
                elif key == 'num_wann':
                    struct[key] = int(words[1])

                elif key == 'dis_win_min':
                    struct[key] = float(words[1])
                elif key == 'dis_win_max':
                    struct[key] = float(words[1])
                elif key == 'dis_froz_min':
                    struct[key] = float(words[1])
                elif key == 'dis_froz_max':
                    struct[key] = float(words[1])

                elif key == 'write_hr':
                    struct[key] = 't' in words[1].lower()
                elif key == 'bands_plot':
                    struct[key] = 't' in words[1].lower()
                elif key == 'wannier_plot':
                    struct[key] = 't' in words[1].lower()

                elif key == 'dis_num_iter':
                    struct[key] = int(words[1])
                elif key == 'num_iter':
                    struct[key] = int(words[1])
                elif key == 'search_shells':
                    struct[key] = int(words[1])

                elif key == 'mp_grid':
                    mp_grid = np.empty(3, dtype=int)
                    for i in range(3):
                        mp_grid[i] = int(words[1 + i])
                    struct[key] = mp_grid

                elif key == 'begin':
                    if words[1] == 'projections':
                        # create sub-dict for projections
                        # complicated solution
                        # not able to save different wannier centres
                        # order of projections not preserved
                        proj_dict = dict()
                        while 'end' not in words:
                            words = next(lines)
                            if 'end' in words:
                                continue
                            else:
                                atom_pos, orbital = words.split(':', 1)
                                proj_dict[atom_pos.strip()] = orbital.strip()

                        struct['proj'] = proj_dict

                    elif words[1] == 'kpoint_path':
                        struct['kpoint_path'] = []

                        while True:
                            line = next(lines).strip()

                            if line.startswith('end'):
                                break

                            struct['kpoint_path'].append(line)

                    elif words[1] == 'unit_cell_cart':
                        struct['unit_cell'] = np.empty((3, 3))

                        for n in range(3):
                            words = next(lines).split()

                            for x in range(3):
                                struct['unit_cell'][n, x] = float(words[x])

                    # read atoms_frac or atoms_cart
                    elif words[1].startswith('atoms_'):
                        struct['atoms_coords'] = words[1][6:len(words[1])]
                        words = next(lines).split()
                        # get nat from lines
                        tmp = []
                        while 'end' != words[0]:
                            tmp.append(words)
                            words = next(lines).split()
                        nat = len(tmp)

                        struct['at'] = []
                        struct['atoms'] = np.empty((nat, 3))

                        for n in range(nat):
                            struct['at'].append(tmp[n][0])
                            for x in range(3):
                                struct['atoms'][n, x] = float(tmp[n][1 + x])

                    elif words[1] == 'kpoints':
                        nk = int(np.prod(mp_grid))
                        struct['kpoints'] = np.zeros((nk, 4))

                        for n in range(nk):
                            words = next(lines).split()

                            for x in range(len(words)):
                                struct['kpoints'][n, x] = float(words[x])
    else:
        struct = None

    struct = comm.bcast(struct)

    return struct

def write_win(win, struct):
    """Write input data to .win file (Wannier90).

    Parameters
    ----------
    win : str
        Filename.
    struct : dict
        Input data.
    """
    if comm.rank != 0:
        return

    with open(win, 'w') as data:
        for key in ['num_bands', 'num_wann']:
            if key in struct:
                data.write('%s = %r\n' % (key, struct[key]))

        data.write('\n')

        for key in ['dis_win_min', 'dis_win_max',
                'dis_froz_min', 'dis_froz_max']:
            if key in struct:
                data.write('%s = %r\n' % (key, struct[key]))

        data.write('\n')

        if struct['proj']:
            data.write('begin projections\n')
            for key, value in struct['proj'].items():
                data.write('%s: %s\n' % (key, value))
            data.write('end projections\n')

        data.write('\n')

        for key in ['write_hr', 'bands_plot']:
            if key in struct:
                data.write('%s = %r\n' % (key, struct[key]))

        data.write('\n')

        for key in ['dis_num_iter', 'num_iter', 'search_shells']:
            if key in struct:
                data.write('%s = %r\n' % (key, struct[key]))

        data.write('\n')

        if struct['kpoint_path']:
            data.write('begin kpoint_path\n')
            for line in struct['kpoint_path']:
                data.write('%s\n' % line)
            data.write('end kpoint_path\n')

        data.write('\n')

        data.write('begin unit_cell_cart\n')
        for (r1, r2, r3) in tuple(struct['unit_cell']):
            data.write('%12.9f %12.9f %12.9f\n' % (r1, r2, r3))
        data.write('end unit_cell_cart\n')

        data.write('\n')

        data.write('begin atoms_%s\n' % struct['atoms_coords'])
        for X, (r1, r2, r3) in zip(struct['at'], struct['atoms']):
            data.write('%2s %12.9f %12.9f %12.9f\n' % (X, r1, r2, r3))
        data.write('end atoms_%s\n' % struct['atoms_coords'])

        data.write('\n')

        if 'mp_grid' in struct:
            data.write('mp_grid: %d %d %d\n' % tuple(struct['mp_grid']))

        data.write('\n')

        data.write('begin kpoints\n')
        for (kx, ky, kz, wk) in struct['kpoints']:
            data.write('%12.9f %12.9f %12.9f %12.9f\n' % (kx, ky, kz, wk))
        data.write('end kpoints\n')

def read_ph(filename):
    """Read input parameters from Quantum ESPRESSO's ``ph.x`` input file.

    Parameters
    ----------
    filename : str
        Filename.

    Returns
    -------
    dict
        Input parameters.
    """
    return elphmod.misc.read_input_data(filename)

def write_ph(ph, struct):
    """Write input data to ph file (Quantum ESPRESSO).

    Parameters
    ----------
    ph : str
        Filename.
    struct : dict
        Input data.
    """
    if comm.rank != 0:
        return

    with open(ph, 'w') as data:
        data.write('&INPUTPH\n')

        for key in ['prefix', 'outdir']:
            if key in struct:
                data.write('%s = %r\n' % (key, struct[key]))

        data.write('\n')

        for key in ['fildyn', 'fildvscf']:
            if key in struct:
                data.write('%s = %r\n' % (key, struct[key]))

        data.write('\n')

        for key in ['ldisp', 'nq1', 'nq2', 'nq3']:
            if key in struct:
                data.write('%s = %r\n' % (key, struct[key]))

        data.write('\n')

        for key in ['tr2_ph', 'alpha_mix', 'cdfpt', 'subspace',
                'cdfpt_bnd', 'cdfpt_orb']:
            if key in struct:
                data.write('%s = %r\n' % (key, struct[key]))

        data.write('/\n')

def read_q2r(filename):
    """Read input parameters from Quantum ESPRESSO's ``q2r.x`` input file.

    Parameters
    ----------
    filename : str
        Filename.

    Returns
    -------
    dict
        Input parameters.
    """
    return elphmod.misc.read_input_data(filename)

def write_q2r(q2r, struct):
    """Write input data to q2r file (Quantum ESPRRESO).

    Parameters
    ----------
    q2r : str
        Filename.
    struct : dict
        Input data.
    """
    if comm.rank != 0:
        return

    with open(q2r, 'w') as data:
        data.write('&INPUT\n')

        for key in ['fildyn', 'flfrc', 'zasr']:
            if key in struct:
                data.write('%s = %r\n' % (key, struct[key]))

        data.write('/\n')

def read_matdyn(filename):
    """Read input parameters from Quantum ESPRESSO's ``matdyn.x`` input file.

    Parameters
    ----------
    filename : str
        Filename.

    Returns
    -------
    dict
        Input parameters.
    """
    struct = elphmod.misc.read_input_data(filename, broadcast=False)

    if comm.rank == 0:
        with open(filename) as lines:
            for line in lines:
                if '/' in line:
                    for line in lines:
                        words = line.split()
                        if words:
                            break

                    struct['nq'] = int(words[0])
                    struct['q'] = np.empty((struct['nq'], 4))

                    for n in range(struct['nq']):
                        words = next(lines).split()

                        for x in range(4):
                            struct['q'][n, x] = float(words[x])

    struct = comm.bcast(struct)

    return struct

def write_matdyn(matdyn, struct):
    """Write input data to matdyn file (Quantum ESPRRESO).

    Parameters
    ----------
    matdyn : str
        Filename.
    struct : dict
        Input data.
    """
    if comm.rank != 0:
        return

    with open(matdyn, 'w') as data:
        data.write('&INPUT\n')

        for key in ['flfrc', 'flfrq', 'fldos', 'fleig', 'flvec', 'asr',
                'loto_2d', 'q_in_band_form', 'q_in_cryst_coord']:
            if key in struct:
                data.write('%s = %r\n' % (key, struct[key]))

        data.write('/\n')
        data.write('%d\n' % struct['nq'])
        for point in struct['q']:
            data.write(' '.join('%12.9f' % q for q in point) + '\n')

def read_epw(filename):
    """Read input parameters from Quantum ESPRESSO's ``epw.x`` input file.

    Parameters
    ----------
    filename : str
        Filename.

    Returns
    -------
    dict
        Input parameters.
    """
    struct = elphmod.misc.read_input_data(filename, broadcast=False)

    if comm.rank == 0:
        for key in ['proj', 'wdata']:
            if key in struct and not isinstance(struct[key], list):
                struct[key] = [struct[key]]

        #with open(filename) as lines:
        #    for line in lines:
        #        if '/' in line:
        #             print(next(lines))
        #             if next(lines) == None:
        #                 continue
        #             words = next(lines).split()
        #             print(words)

        #             struct['nq'] = int(words[0])
        #             struct['q_coords_type'] = words[1]

        #             struct['q'] = np.empty((struct['nq'], 3))

        #             for n in range(struct['nq']):
        #                 words = next(lines).split()

        #                 for x in range(3):
        #                     struct['q'][n, x] = float(words[x])

    struct = comm.bcast(struct)

    return struct

def write_epw(epw, struct):
    """Write input data to epw file (Quantum ESPRRESO).

    Parameters
    ----------
    epw : str
        Filename.
    struct : dict
        Input data.
    """
    if comm.rank != 0:
        return

    with open(epw, 'w') as data:
        data.write('&INPUTEPW\n')

        for key in ['prefix', 'outdir', 'dvscf_dir']:
            if key in struct:
                data.write('%s = %r\n' % (key, struct[key]))

        data.write('\n')

        for key in ['wannierize', 'elph']:
            if key in struct:
                data.write('%s = %r\n' % (key, struct[key]))

        data.write('\n')

        for key in ['epbwrite', 'epwwrite']:
            if key in struct:
                data.write('%s = %r\n' % (key, struct[key]))

        data.write('\n')

        for key in ['epbread', 'epwread']:
            if key in struct:
                data.write('%s = %r\n' % (key, struct[key]))

        data.write('\n')

        for key in ['use_ws', 'nbndsub', 'bands_skipped']:
            if key in struct:
                data.write('%s = %r\n' % (key, struct[key]))

        data.write('\n')

        for key in ['dis_win_min', 'dis_win_max',
                'dis_froz_min', 'dis_froz_max']:
            if key in struct:
                data.write('%s = %r\n' % (key, struct[key]))

        data.write('\n')

        if 'proj' in struct:
            for n, value in enumerate(struct['proj'], 1):
                if value:
                    data.write('proj(%d) = %r\n' % (n, value))

        data.write('\n')

        for key in ['num_iter']:
            if key in struct:
                data.write('%s = %r\n' % (key, struct[key]))

        if 'wdata' in struct:
            for n, value in enumerate(struct['wdata'], 1):
                if value:
                    data.write('wdata(%d) = %r\n' % (n, value))

        data.write('\n')

        for key in ['nk1', 'nk2', 'nk3']:
            if key in struct:
                data.write('%s = %r\n' % (key, struct[key]))

        data.write('\n')

        for key in ['nq1', 'nq2', 'nq3']:
            if key in struct:
                data.write('%s = %r\n' % (key, struct[key]))

        data.write('\n')

        for key in ['nkf1', 'nkf2', 'nkf3']:
            if key in struct:
                data.write('%s = %r\n' % (key, struct[key]))

        data.write('\n')

        for key in ['nqf1', 'nqf2', 'nqf3']:
            if key in struct:
                data.write('%s = %r\n' % (key, struct[key]))

        data.write('/')
        data.write('\n')
# =============================================================================
#         data.write('%d %s\n' % (struct['nq'], struct['q_coords_type']))
#         for (qx, qy, qz) in struct['q']:
#             data.write('%12.9f %12.9f %12.9f\n' % (qx, qy, qz))
# =============================================================================

def readPOSCAR(filename):
    """Read crystal structure from VASP input file.

    Parameters
    ----------
    filename : str
        Filename.

    Returns
    -------
    dict
        Crystal structure.
    """
    with open(filename) as POSCAR:
        title = next(POSCAR)

        a = float(next(POSCAR))

        t1 = np.array(list(map(float, next(POSCAR).split())))
        t2 = np.array(list(map(float, next(POSCAR).split())))
        t3 = np.array(list(map(float, next(POSCAR).split())))

        elements = next(POSCAR).split()

        numbers = list(map(int, next(POSCAR).split()))

        next(POSCAR)
        next(POSCAR)

        atoms = dict()

        for element, number in zip(elements, numbers):
            atoms[element] = np.empty((number, 3))

            for n in range(number):
                atoms[element][n] = list(map(float, next(POSCAR).split()[:3]))

    return t1, t2, t3, atoms

def point_on_path(test_point, point_A, point_B, eps=1e-14):
    """Test whether a `test_point` is between the points A and B.

    Parameters
    ----------
    eps : float
        Numerical parameter, in case the cross product is not exactly 0.

    Returns
    -------
    bool
        Is the test_point on a straight line between point A and B?
    """
    cross = np.cross(point_B - point_A, test_point - point_A)
    if all(abs(v) < eps for v in cross):
        dot = np.dot(point_B - point_A, test_point - point_A)

        if dot >= 0:
            squared_distance = (
                (point_B - point_A)[0] ** 2 + (point_B - point_A)[1] ** 2)

            if dot <= squared_distance: # test point between A and B
                return True

def crystal_to_cartesian(R_CRYSTAL, a1, a2, a3=None):
    """Transform a lattice structure from crystal to Cartesian coordinates.

    Parameters
    ----------
    R_CRYSTAL : ndarray
        Lattice structure in crystal coordinates.
    a1, a2, a3 : ndarray
        Lattice vectors.

    Returns
    -------
    R_CARTESIAN : ndarray
        Lattice structure in Cartesian coordinates.
    """
    R_CARTESIAN = np.empty(R_CRYSTAL.shape)

    if a3 is None:
        for ii in np.arange(R_CARTESIAN.shape[0]):
            R_CARTESIAN[ii, :] = R_CRYSTAL[ii, 0] * a1 + R_CRYSTAL[ii, 1] * a2
    else:
        for ii in np.arange(R_CARTESIAN.shape[0]):
            R_CARTESIAN[ii, :] = (R_CRYSTAL[ii, 0] * a1 + R_CRYSTAL[ii, 1] * a2
                + R_CRYSTAL[ii, 2] * a3)

    return R_CARTESIAN

def cartesian_to_crystal(R_CARTESIAN, a1, a2, a3):
    """Transform a lattice structure from crystal to Cartesian coordinates.

    Parameters
    ----------
    R_CARTESIAN : ndarray
        Lattice structure in Cartesian coordinates.
    a1, a2, a3 : ndarray
        Lattice vectors.

    Returns
    -------
    R_CRYSTAL: ndarray
        Lattice structure in crystal coordinates.
    """
    R_CRYSTAL = np.empty(R_CARTESIAN.shape)

    A_Matrix = np.zeros([3, 3])

    A_Matrix[:, 0] = a1
    A_Matrix[:, 1] = a2
    A_Matrix[:, 2] = a3

    A_Matrix_Inverse = np.linalg.inv(A_Matrix)

    for ii in np.arange(R_CARTESIAN.shape[0]):
        R_CRYSTAL[ii, :] = np.dot(A_Matrix_Inverse, R_CARTESIAN[ii, :])

    return R_CRYSTAL

def mesh(*n, flat=False):
    r"""Generate uniform mesh.

    Parameters
    ----------
    *n : int
        Mesh dimensions.
    flat : bool, default False
        Flatten mesh-point indices into single dimension?

    Returns
    -------
    ndarray
        Mesh points with period :math:`2 \pi`.
    """
    points = 2 * np.pi * np.moveaxis(np.array(np.meshgrid(*map(range, n),
        indexing='ij'), dtype=float), 0, -1) / n

    if flat:
        points = points.reshape((-1, len(n)))

    return points.copy()

def kpoints(nk1=None, nk2=None, nk3=None, weights=None):
    """Generate and print uniform k-point mesh.

    Omitted arguments are read from standard input.

    Parameters
    ----------
    nk1, nk2, nk3 : int
        Number of points along axis.
    weights : bool
        Print weights?
    """
    nk = [n or int(input('number of points along %s axis: ' % axis) or 1)
        for axis, n in [('1st', nk1), ('2nd', nk2), ('3rd', nk3)]]

    N = np.prod(nk)

    if weights is None:
        weights = 'y' in input('print weights (y/n): ')

    w = (1 / N,) * weights

    print(N)

    for k in mesh(*nk).reshape((-1, 3)) / (2 * np.pi):
        print(' '.join('%12.10f' % c for c in tuple(k) + w))
