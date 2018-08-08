#/usr/bin/env python

from __future__ import division

import numpy as np

from . import MPI
comm = MPI.comm

deg = np.pi / 180

def rotate(vector, angle):
    """Rotate vector."""

    return np.dot(np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)],
        ]), vector)

def translations(angle=120, angle0=0):
    """Get translation vectors of Bravais lattice."""

    t1 = np.array([1.0, 0.0])

    t1 = rotate(t1, angle0 * deg)
    t2 = rotate(t1, angle  * deg)

    return t1, t2

def reciprocals(t1, t2):
    """Get translation vectors of reciprocal lattice (w/o 2 pi)."""

    u1 = rotate(t2, -90 * deg)
    u2 = rotate(t1, +90 * deg)

    u1 /= np.dot(t1, u1)
    u2 /= np.dot(t2, u2)

    return u1, u2

def images(k1, k2, nk, angle=60):
    """Get equivalents k points."""

    points = set()

    for _ in range(6):
        # rotation by 60 degree:

        if angle == 60:
            k1, k2 = -k2, k1 + k2
        elif angle == 120:
            k1, k2 = k1 - k2, k1

        # mapping to [0, nk):

        k1 %= nk
        k2 %= nk

        # add point and its reflection:

        points.add((k1, k2))
        points.add((k2, k1))

    return points

def irreducibles(nk, angle=60):
    """Get irreducible k points."""

    points = [
        (k1, k2)
        for k1 in range(nk)
        for k2 in range(nk)]

    irreducible = set(points)

    for k in points:
        if k in irreducible:
            reducible = images(*k, nk=nk, angle=angle)
            reducible.discard(k)
            irreducible -= reducible

    return irreducible

def symmetries(data, epsilon=0.0, unity=True, angle=60):
    """Find symmetries of data on Monkhorst-Pack mesh."""

    t1, t2, u1, u2 = vectors(angle, angle0=0)
    # t1 and u2 must point in x and y direction, respectively,
    # to make below reflection work properly.

    nk = len(data)

    def get_image(reflect, angle):
        image = np.empty((nk, nk, 2), dtype=int)

        for k1 in range(nk):
            for k2 in range(nk):
                K = rotate(k1 * u1 + k2 * u2, angle * deg)

                if reflect:
                    K[0] *= -1

                K1 = int(round(np.dot(K, t1))) % nk
                K2 = int(round(np.dot(K, t2))) % nk

                if abs(data[k1, k2] - data[K1, K2]) > epsilon:
                    return None

                image[k1, k2] = (K1, K2)

        return image

    for reflect in False, True:
        for angle in range(0, 360, 60):
            if reflect or angle or unity:
                image = get_image(reflect, angle)

                if image is not None:
                    yield (reflect, angle), image

def complete(data, angle=60):
    """Complete data on Monkhorst-Pack mesh."""

    irreducible = list(zip(*np.where(np.logical_not(np.isnan(data)))))

    if len(irreducible) == data.size:
        return

    for symmetry, image in symmetries(data, unity=False, angle=angle):
        for k in irreducible:
            data[tuple(image[k])] = data[k]

        if not np.isnan(data).any():
            return

def linear_interpolation(data, angle=60, axes=(0, 1)):
    """Perform linear interpolation on triangular or rectangular lattice."""

    order = tuple(axes) + tuple(n for n in range(data.ndim) if n not in axes)

    data = np.transpose(data, axes=order)

    N, M = data.shape[:2]

    def split(n, m):
        n0, dn = divmod(n, 1)
        m0, dm = divmod(m, 1)
        n0 = int(n0) % N
        m0 = int(m0) % M

        return (n0, dn), (m0, dm)

    if angle == 60:
        def interpolant(n, m):
            (n0, dn), (m0, dm) = split(n, m)

            A = data[(n0 + 1) % N, m0]
            B = data[n0, (m0 + 1) % M]

            if dn + dm > 1:
                C = data[(n0 + 1) % N, (m0 + 1) % M]
                return (1 - dm) * A + (1 - dn) * B + (dn + dm - 1) * C
            else:
                C = data[n0, m0]
                return dn * A + dm * B + (1 - dn - dm) * C

    elif angle == 90:
        def interpolant(n, m):
            (n0, dn), (m0, dm) = split(n, m)

            A = data[n0, m0]
            B = data[(n0 + 1) % N, m0]
            C = data[(n0 + 1) % N, (m0 + 1) % M]
            D = data[n0, (m0 + 1) % M]

            return ((1 - dn) * (1 - dm) * A +      dn  * (1 - dm) * B
                +        dn  *      dm  * C + (1 - dn)      * dm  * D)

    elif angle == 120:
        def interpolant(n, m):
            (n0, dn), (m0, dm) = split(n, m)

            A = data[n0, m0]
            B = data[(n0 + 1) % N, (m0 + 1) % M]

            if dn > dm:
                C = data[(n0 + 1) % N, m0]
                return (1 - dn) * A + dm * B + (dn - dm) * C
            else:
                C = data[n0, (m0 + 1) % M]
                return (1 - dm) * A + dn * B + (dm - dn) * C

    return np.vectorize(interpolant)

def resize(data, shape=None, angle=60, axes=(0, 1)):
    """Resize array via linear interpolation along two periodic axes."""

    order = tuple(axes) + tuple(n for n in range(data.ndim) if n not in axes)

    data = np.transpose(data, axes=order)

    interpolant = linear_interpolation(data, angle, axes=(0, 1))

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

    new_data = np.transpose(new_data, axes=np.argsort(order)) # restore order

    return new_data

def MPM2IBZ(k1, k2, nk, angle=60):
    """Map from positive Monkhorst-Pack mesh to first Brillouin zone."""

    # squared distance from origin:

    measure = {
         60: lambda n, m: n * n + m * m + n * m,
         90: lambda n, m: n * n + m * m,
        120: lambda n, m: n * n + m * m - n * m,
        }

    # For 60 or 120 deg. (triangular lattice) this yields the Loeschian numbers.
    # Non-equivalent lattice sites may have the same distance from the origin!
    # (E.g., there are non-equivalent 20th neighbors in a triangular lattice)

    images = [(k1, k2), (k1 - nk, k2), (k1, k2 - nk), (k1 - nk, k2 - nk)]
    distances = [measure[angle](*image) for image in images]
    minimum = min(distances)
    images = [image for image, distance in zip(images, distances)
        if distance == minimum]

    return images

def Fourier_interpolation(data, angle=60, hr_file=None, function=True):
    """Perform Fourier interpolation on triangular or rectangular lattice."""

    N, N = data.shape

    # do first Fourier transform to obtain coefficients:

    i = np.arange(N)

    transform = np.exp(2j * np.pi / N * np.outer(i, i)) / N

    data = np.dot(np.dot(transform, data), transform)

    # construct smooth inverse transform (formally tight-binding model):

    values = np.empty((N * N * 4), dtype=complex)
    points = np.empty((N * N * 4, 2), dtype=int)
    counts = np.empty((N * N * 4), dtype=int)

    count = 0
    for n in range(N):
        for m in range(N):
            images = MPM2IBZ(n, m, N, angle=180 - angle)

            # angle transform: from real to reciprocal lattice or vice versa

            for point in images:
                values[count] = data[n, m]
                points[count] = point
                counts[count] = len(images)
                count += 1

    values = values[:count]
    points = points[:count]
    counts = counts[:count]

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

    values /= counts

    idphi = -2j * np.pi / N

    def interpolant(*point):
        return np.dot(values, np.exp(idphi * np.dot(points, point))).real

    if function:
        return np.vectorize(interpolant)

    return dict((tuple(point), value) for point, value in zip(points, values))

def GMKG(N=30, corner_indices=False, mesh=False, angle=60):
    """Generate path Gamma-M-K-Gamma through Brillouin zone."""

    if angle == 60:
        K1 = 1.0
    elif angle == 120:
        K1 = 2.0

    G = 2 * np.pi * np.array([0.0, 0.0])
    M = 2 * np.pi * np.array([0.0, 1.0]) / 2
    K = 2 * np.pi * np.array([1.0, K1 ]) / 3

    L1 = np.sqrt(3)
    L2 = 1.0
    L3 = 2.0

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
         + line(K, G, N3, True)

    x = np.empty(N1 + N2 + N3)

    x[      0:N1          ] = np.linspace(      0, L1,           N1, False)
    x[     N1:N1 + N2     ] = np.linspace(     L1, L1 + L2,      N2, False)
    x[N2 + N1:N1 + N2 + N3] = np.linspace(L2 + L1, L1 + L2 + L3, N3, True)

    if corner_indices:
        return np.array(path), x, [0, N1, N1 + N2, N1 + N2 + N3 - 1]
    else:
        return np.array(path), x
