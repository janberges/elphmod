#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

from __future__ import division

import numpy as np
import sys

from . import MPI
comm = MPI.comm

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
        a2 = rotate(a1, angle  * deg)

        return a1, a2
    else:
        a1 = np.array([1.0, 0.0, 0.0])

        a1 = rotate(a1, angle0 * deg, two_dimensional=False)
        a2 = rotate(a1, angle  * deg, two_dimensional=False)

        return a1, a2

def reciprocals(a1, a2, a3=None):
    """Generate translation vectors of reciprocal lattice.

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
    """Generate set of irreducible k points.

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
        Maxmium absolute difference of "equal" floats.
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
                # rotation in cartesian coordinates:

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

    for reflect in False, True:
        for angle in range(0, 360, 60):
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

    # map points onto interval [0, period]:

    points = np.array(points) % period

    # generate list of indices that would sort points:

    order = np.argsort(points)

    # save "stacking", shift lowest point up by one period, and repeat:

    stackings = np.empty((points.size, points.size), dtype=points.dtype)

    stackings[0] = points

    for n in range(points.size - 1):
        stackings[n + 1] = stackings[n]
        stackings[n + 1, order[n]] += period

    # return most localized stacking:

    return min(stackings, key=np.std)


def linear_interpolation(data, angle=60, axes=(0, 1), period=None):
    """Perform linear interpolation on triangular or rectangular lattice.

    The edges are interpolated using periodic boundary conditions.

    Parameters
    ----------
    data : ndarray
        Data on uniform triangular or rectangular lattice.
    angle : number
        Angle between lattice vectors in degrees.
    axes : 2-tuple of ints
        Axes of `data` along which to interpolate (lattice vectors).
    period : number
        If the values of `data` are defined on a periodic axis (i.e., only with
        respect to the modulo operation), the period of this axis. This is used
        in combination with `stack` to always interpolate across the shortest
        distance of two neighbording points.

    Returns
    -------
    function
        Interpolant for `data`. ``linear_interpolation(data)(i, j)`` yields the
        same value as ``data[i, j]``. Thus the data array is "generalized" with
        respect to fractional indices.

    See Also
    --------
    stack : Condese point cloud on periodic axis.
    resize : Compress or strech data via linear interpolation.
    Fourier_interpolation : Alternative interpolation routine.
    """
    # move lattice axes to the front:

    order = tuple(axes) + tuple(n for n in range(data.ndim) if n not in axes)

    data = np.transpose(data, axes=order)

    # interpret "fractional indices":

    N, M = data.shape[:2]

    def split(n, m):
        n0, dn = divmod(n, 1)
        m0, dm = divmod(m, 1)
        n0 = int(n0) % N
        m0 = int(m0) % M

        return (n0, dn), (m0, dm)

    # define interpolation routines for different lattices:

    if angle == 60:
        #
        #     B______C'
        #     /\    /
        # a2 /  \  /
        #   /____\/
        #  C  a1  A
        #
        def interpolant(n, m):
            (n0, dn), (m0, dm) = split(n, m)

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
            (n0, dn), (m0, dm) = split(n, m)

            A = data[n0, m0]
            B = data[(n0 + 1) % N, m0]
            C = data[(n0 + 1) % N, (m0 + 1) % M]
            D = data[n0, (m0 + 1) % M]

            if period:
                A, B, C, D = stack(A, B, C, D, period=period)

            return ((1 - dn) * (1 - dm) * A +      dn  * (1 - dm) * B
                +        dn  *      dm  * C + (1 - dn)      * dm  * D)

    elif angle == 120:
        #
        #  C______B
        #   \    /\
        # a2 \  /  \
        #     \/____\
        #     A  a1  C'
        #
        def interpolant(n, m):
            (n0, dn), (m0, dm) = split(n, m)

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

def resize(data, shape=None, angle=60, axes=(0, 1), period=None):
    """Resize array via linear interpolation along two periodic axes.

    Parameters
    ----------
    shape : 2-tuple of ints
        New lattice shape. Defaults to the original shape.
    shape, angle, axes, period
        Parameters for :func:`linear_interpolation`.

    Returns
    -------
    ndarray
        Resized data array.

    See Also
    --------
    linear_interpolation
    """
    # move lattice axes to the front:

    order = tuple(axes) + tuple(n for n in range(data.ndim) if n not in axes)

    data = np.transpose(data, axes=order)

    # set up interpolation function:

    interpolant = linear_interpolation(data, angle, axes=(0, 1), period=period)

    # apply interpolation function at new lattice points in parallel:

    if shape is None:
        shape = data.shape[:2]

    size = np.prod(shape)
    sizes, bounds = MPI.distribute(size, bounds=True)

    my_new_data = np.empty((sizes[comm.rank],) + data.shape[2:],
        dtype=data.dtype)

    scale_x = data.shape[0] / shape[0]
    scale_y = data.shape[1] / shape[1]

    for n, m in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        x = m // shape[1]
        y = m %  shape[1]

        my_new_data[n] = interpolant(x * scale_x, y * scale_y)

    new_data = np.empty(tuple(shape) + data.shape[2:], dtype=data.dtype)

    comm.Allgatherv(my_new_data, (new_data, sizes * np.prod(data.shape[2:])))

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
    k1, k2 : integer
        Point in lattice coordinates (crystal coordinates, mesh-indices, ...).
    angle : number
        Angle between lattice axes.

    Returns
    -------
    number
        Squared distance of point from origin.
    """
    sgn = { 60: 1, 90: 0, 120: -1 }[angle]

    return k1 * k1 + k2 * k2 + sgn * k1 * k2

def to_Voronoi(k1, k2, nk, angle=60, dk1=0, dk2=0, epsilon=0.0):
    """Map any lattice point to the Voronoi cell* around the origin.

    (*) Wigner-Seitz cell/Brillouin zone for Bravais/reciprocal lattice.

    Parameters
    ----------
    k1, k2 : integer
        Mesh-point indices.
    nk : int
        Number of points per dimension.
    angle : number
        Angle between lattice vectors.
    dk1, dk2 : number
        Shift of Voronoi cell.
    epsilon : float
        Maxmium absolute difference of "equal" floats.

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

def wigner_seitz(nk, angle=120, dk1=0.0, dk2=0.0, epsilon=0.0):
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
        Maxmium absolute difference of "equal" floats.

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

def wigner_seitz_x(x, nk, at=None, tau=None, epsilon=1e-8):
    """Emulate the EPW subroutine *wigner_seitz{x}* in *wigner.f90*.

    Parameters
    ----------
    x : str
        Type of Wigner-Seitz cell:

        * ``'k'``: cell-centered
        * ``'q'``: bond-centered
        * ``'g'``: atom-centered

    nk : int
        Number of points per dimension.
    at, tau : ndarray
        Geometry as returned by :func:`ph.read_flfrc` and :func:`ph.model`.
    epsilon : float
        Maxmium absolute difference of "equal" floats.

    Returns
    -------
    list of tuple of int
        Mesh-point indices.
    list of int
        Degeneracies.
    list of float
        Lattice-vector lengths.
    """
    a = np.sqrt(np.dot(at[0, :2], at[0, :2]))

    a1 = at[0, :2] / a
    a2 = at[1, :2] / a

    angle = int(round(np.arccos(np.dot(a1, a2)) * 180 / np.pi))

    b1, b2 = reciprocals(a1, a2)

    if x == 'k':
        return wigner_seitz(nk, angle)

    if x == 'g':
        shifts = tau

    elif x == 'q':
        shifts = [tau2 - tau1 for tau1 in tau for tau2 in tau]

    irvec_x  = []
    ndegen_x = [] # list of dict
    wslen_x  = dict()

    for dk in shifts:
        dk1 = np.dot(b1, dk[:2]) / a
        dk2 = np.dot(b2, dk[:2]) / a

        irvec, ndegen, wslen = wigner_seitz(nk, angle, -dk1, -dk2, epsilon)

        irvec_x.extend([key for key in irvec if key not in wslen_x])

        ndegen_x.append(dict(zip(irvec, ndegen)))
        wslen_x.update(dict(zip(irvec, wslen)))

    ndegen_x = [[ndegen.get(key, 0) for key in irvec_x] for ndegen in ndegen_x]
    wslen_x = [wslen_x[key] for key in irvec_x]

    if x == 'q':
        ndegen_x = np.reshape(ndegen_x, (len(tau), len(tau), len(irvec_x)))
        ndegen_x = np.transpose(ndegen_x, axes=(1, 0, 2))

    return irvec_x, ndegen_x, wslen_x

def write_wigner_file(name, nk, nq, at=None, tau=None, epsilon=1e-8):
    """Write binary file with Wigner-Seitz data as used by EPW.

    **ATTENTION:** This function is compatible with `read_wigner_file` for
    ``old_ws=True`` only!

    See Also
    --------
    read_wigner_file, wigner_seitz_x, elph.Model
    """
    if comm.rank == 0:
        integer = np.int32
        double  = np.float64

        with open(name, 'wb') as data:
            for x, nx in zip('kqg', [nk, nq, nq]):
                irvec, ndegen, wslen = wigner_seitz_x(x, nx, at, tau, epsilon)

                irvec = np.insert(irvec, obj=2, values=0, axis=1) # 2D to 3D

                np.array(len(irvec), dtype=integer).tofile(data)
                np.array(    irvec,  dtype=integer).tofile(data)
                np.array(   ndegen,  dtype=integer).tofile(data)
                np.array(    wslen,  dtype=double ).tofile(data)

def read_wigner_file(name, old_ws=False, nat=None):
    """Read binary file with Wigner-Seitz data as used by EPW.

    Parameters
    ----------
    name : str
        Name of file with Wigner-Seitz data.
    old_ws : bool
        Use previous definition of Wigner-Seitz cells?
    nat : int
        Number of atoms per unit cell.

    See Also
    --------
    write_wigner_file, wigner_seitz_x, elph.Model
    """
    if comm.rank == 0:
        with open(name, 'rb') as data:
            integer = np.int32
            double  = np.float64

            if old_ws:
                nrr_k    = np.fromfile(data, integer, 1)[0]
                irvec_k  = np.fromfile(data, integer, nrr_k * 3)
                irvec_k  = irvec_k.reshape((nrr_k, 3))
                ndegen_k = np.fromfile(data, integer, nrr_k)
                wslen_k  = np.fromfile(data, double, nrr_k)

                nrr_q    = np.fromfile(data, integer, 1)[0]
                irvec_q  = np.fromfile(data, integer, nrr_q * 3)
                irvec_q  = irvec_q.reshape((nrr_q, 3))
                ndegen_q = np.fromfile(data, integer, nat * nat * nrr_q)
                ndegen_q = ndegen_q.reshape((nat, nat, nrr_q))
                wslen_q  = np.fromfile(data, double, nrr_q)

                nrr_g    = np.fromfile(data, integer, 1)[0]
                irvec_g  = np.fromfile(data, integer, nrr_g * 3)
                irvec_g  = irvec_g.reshape((nrr_g, 3))
                ndegen_g = np.fromfile(data, integer, nat * nrr_g)
                ndegen_g = ndegen_g.reshape((nat, nrr_g))
                wslen_g  = np.fromfile(data, double, nrr_g)

            else:
                nbndsub, = np.fromfile(data, integer, 1)
                nat,     = np.fromfile(data, integer, 1)

                nrr_k, = np.fromfile(data, integer, 1)
                irvec_k = np.fromfile(data, integer, nrr_k * 3)
                irvec_k = irvec_k.reshape((nrr_k, 3))
                ndegen_k = np.fromfile(data, integer, nbndsub ** 2 * nrr_k)
                ndegen_k = ndegen_k.reshape((nbndsub, nbndsub, nrr_k))

                nrr_g, = np.fromfile(data, integer, 1)
                irvec_g = np.fromfile(data, integer, nrr_g * 3)
                irvec_g = irvec_g.reshape((nrr_g, 3))
                ndegen_g = np.fromfile(data, integer, nbndsub ** 2 * nat * nrr_g)
                ndegen_g = ndegen_g.reshape((nbndsub, nbndsub, nat, nrr_g))

        data = irvec_k, ndegen_k, irvec_g, ndegen_g
    else:
        data = None

    data = comm.bcast(data)

    return data

def Fourier_interpolation(data, angle=60, sign=-1, hr_file=None, function=True):
    """Perform Fourier interpolation on triangular or rectangular lattice.

    Parameters
    ----------
    data : ndarray
        Data on uniform triangular or rectangular lattice.
    angle : number
        Angle between lattice vectors in degrees.
    sign : number
        Sign in exponential function in first Fourier transform.
    hr_file : str
        Filename. If given, save *_hr.dat* file as produced by Wannier90.
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
    N, N = data.shape

    # do first Fourier transform to obtain coefficients:

    i = np.arange(N)

    transform = np.exp(sign * 2j * np.pi / N * np.outer(i, i)) / N

    data = np.dot(np.dot(transform, data), transform)

    # construct smooth inverse transform (formally tight-binding model):

    values = np.empty((N * N * 4), dtype=complex)
    points = np.empty((N * N * 4, 2), dtype=int)
    counts = np.empty((N * N * 4), dtype=int)

    count = 0
    for n in range(N):
        for m in range(N):
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

    # write "tight-binding model" to disk:

    if hr_file is not None:
        import time

        order = np.lexsort((points[:,1], points[:,0]))

        with open(hr_file, 'w') as hr:
            hr.write(time.strftime(' written on %d%b%Y at %H:%M:%S\n'))

            hr.write('%12d\n' % 1)
            hr.write('%12d\n' % count)

            columns = 15

            for n, i in enumerate(order, 1):
                hr.write('%5d' % counts[i])

                if not n % columns or n == count:
                    hr.write('\n')

            form = '%5d' * 5 + '%12.6f' * 2 + '\n'

            for i in order:
                hr.write(form % (points[i, 0], points[i, 1], 0, 1, 1,
                    values[i].real, values[i].imag))

    # fix weights of interpolation coefficients:

    values /= counts

    # define interpolation function and generalize is with respect to arrays:

    idphi = -sign * 2j * np.pi / N

    def interpolant(*point):
        return np.dot(values, np.exp(idphi * np.dot(points, point))).real

    if function:
        return np.vectorize(interpolant)

    # return either interpolation function or parameter dictionary:

    return dict((tuple(point), value) for point, value in zip(points, values))

def path(points, b, N=30):
    """Generate arbitrary path through Brillouin zone.

    Parameters
    ----------
    points : ndarray
        List of high-symmetry points in crystal coordinates.
    b : ndarray
        List of reciprocal lattice vectors.
    N : integer
        Number of points per :math:`2 \pi / a`.

    Returns
    -------
    ndarray
        Points in crystal coordinates with period :math:`2 \pi`.
    ndarray
        Cumulative path distance.
    list, optional
        Indices of corner/high-symmetry points.
    """
    points = np.array(points)

    points_cart = np.einsum('kc,cx->kx', points, b)

    k = []
    x = []
    corners = [0]

    x0 = 0.0

    for i in range(len(points) - 1):
        dx = np.linalg.norm(points_cart[i + 1] - points_cart[i])
        n = int(round(N * dx))

        x1 = x0 + dx

        for j in range(0 if i == 0 else 1, n + 1):
            k.append((j * points[i + 1] + (n - j) * points[i]) / n)
            x.append((j * x1 + (n - j) * x0) / n)

        x0 = x1

        corners.append(len(k) - 1)

    return 2 * np.pi * np.array(k), np.array(x), corners

def GMKG(N=30, corner_indices=False, mesh=False, angle=60, straight=True,
        lift_degen=True):
    """Generate path |Ggr|-M-K-|Ggr| through Brillouin zone.

    (This function should be replaced by a more general one to produce arbitrary
    paths through both triangular and rectangular Brillouin zones, where special
    points can be defined using labels such as ``'G'``, ``'M'``, or ``'K'``.)

    Parameters
    ----------
    N : integer
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

    L1 = 1.0 / np.sqrt(3)
    L2 = 1.0 / 3.0
    L3 = 2.0 / 3.0

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

    x[      0:N1          ] = np.linspace(      0, L1,           N1, False)
    x[     N1:N1 + N2     ] = np.linspace(     L1, L1 + L2,      N2, False)
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
    """Read crystal structure from PW input file.

    Parameters
    ----------
    pwi : str
        File name.

    Returns
    -------
    dict
        Crystal structure.
    """
    struct = dict()

    if comm.rank == 0:
        with open(pwi) as lines:
            for line in lines:
                words = line.split()

                if not words:
                    continue

                key = words[0].lower()

                if key in 'abc':
                    struct[key] = float(words[2])

                elif key == 'nat':
                    struct[key] = int(words[2])

                elif key == 'atomic_positions':
                    struct['at'] = []
                    struct['r'] = np.empty((struct['nat'], 3))

                    for n in range(struct['nat']):
                        words = next(lines).split()

                        struct['at'].append(words[0])

                        for x in range(3):
                            struct['r'][n, x] = float(words[1 + x])
    else:
        struct = None

    struct = comm.bcast(struct)

    return struct

def write_pwi(pwi, struct):
    """Write crystal structure to PW input file.

    Parameters
    ----------
    pwi : str
        File name.
    struct : dict
        Crystal structure.
    """
    if comm.rank != 0:
        return

    with open(pwi, 'w') as data:
        data.write('&SYSTEM\n')

        for key in 'a', 'b', 'c', 'nat':
            if key in struct:
                data.write('%3s = %.10g\n' % (key, struct[key]))

        data.write('/\n')

        data.write('ATOMIC_POSITIONS\n')

        for X, (r1, r2, r3) in zip(struct['at'], struct['r']):
            data.write('%2s %12.9f %12.9f %12.9f\n' % (X, r1, r2, r3))

        data.write('/\n')

def readPOSCAR(filename):
    """Read crystal structure from VASP input file.

    Parameters
    ----------
    filename : str
        File name.

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


def point_on_path(test_point, point_A, point_B, eps = 1e-14):
    """Test wether a test_point is between the points A and B.

    Parameters
    ----------
    eps: float
    Numerical parameter, in case the the cross product is not exactly 0.

    Returns
    -------
    bool
        True, if the test_point is on a straight line between point A and B
    """

    cross = np.cross(point_B-point_A, test_point-point_A)
    if all(abs(v) < eps  for v in cross):
        dot = np.dot((point_B-point_A), test_point-point_A)

        if dot>=0:

            squared_distance = (point_B-point_A)[0]**2+(point_B-point_A)[1]**2

            if dot<=squared_distance:
                #'The test point is between A and B'
                return True


def crystal_to_cartesian(R_CRYSTAL, a1,a2,a3):
    """Transform a lattice structure R_CRYSTAL from crystal coordinates to
    cartesian coordinates.

    Parameters
    ----------
    R_CRYSTAL: ndarray
        Lattice structure in crystal coordinates
    a1,a2,a3: ndarray
        Lattice vectors

    Returns
    -------
    R_CARTESIAN: ndarray
        Lattice structure in cartesian coordinates
    """
    R_CARTESIAN = np.empty(R_CRYSTAL.shape)

    for ii in np.arange(R_CARTESIAN.shape[0]):
        R_CARTESIAN[ii,:] = R_CRYSTAL[ii,0]*a1+R_CRYSTAL[ii,1]*a2+R_CRYSTAL[ii,2]*a3
    return R_CARTESIAN

def cartesian_to_crystal(R_CARTESIAN, a1,a2,a3):
    """Transform a lattice structure R_CARTESIAN from crystal coordinates to
    cartesian coordinates.

    Parameters
    ----------
    R_CARTESIAN: ndarray
        Lattice structure in cartesian coordinates
    a1,a2,a3: ndarray
        Lattice vectors

    Returns
    -------
    R_CRYSTAL: ndarray
        Lattice structure in crystal coordinates
    """
    R_CRYSTAL = np.empty(R_CARTESIAN.shape)
    A_Matrix = np.zeros([3,3])

    A_Matrix[:,0] = a1
    A_Matrix[:,1] = a2
    A_Matrix[:,2] = a3

    A_Matrix_Inverse = np.linalg.inv(A_Matrix)

    for ii in np.arange(R_CARTESIAN.shape[0]):
        R_CRYSTAL[ii,:] = np.dot(A_Matrix_Inverse, R_CARTESIAN[ii,:])
    return R_CRYSTAL

