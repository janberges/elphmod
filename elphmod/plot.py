# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

"""BZ plots, fatbands, etc."""

import numpy as np

import elphmod.bravais
import elphmod.MPI
import elphmod.misc

comm = elphmod.MPI.comm

def choose_backend():
    """Switch to non-GUI Matplotlib backend if necessary."""

    import os
    import matplotlib

    if 'DISPLAY' not in os.environ:
        matplotlib.use('agg')

def plot(mesh, kxmin=-1.0, kxmax=1.0, kymin=-1.0, kymax=1.0, resolution=100,
        interpolation=elphmod.bravais.linear_interpolation, angle=60,
        return_k=False, broadcast=True):
    """Plot in Cartesian reciprocal coordinates."""

    nk, nk = mesh.shape

    nkx = int(round(resolution * (kxmax - kxmin)))
    nky = int(round(resolution * (kymax - kymin)))

    # (note that endpoint=False is combined with a retstep/2 shift)
    kx, dkx = np.linspace(kxmin, kxmax, nkx, endpoint=False, retstep=True)
    ky, dky = np.linspace(kymin, kymax, nky, endpoint=False, retstep=True)

    kx += dkx / 2
    ky += dky / 2

    ky = ky[::-1]

    fun = interpolation(mesh, angle=angle)

    a1, a2 = elphmod.bravais.translations(180 - angle)

    sizes, bounds = elphmod.MPI.distribute(nky * nkx, bounds=True)

    my_image = np.empty(sizes[comm.rank], dtype=mesh.dtype)

    status = elphmod.misc.StatusBar(sizes[comm.rank], title='plot')

    for n, m in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        i = m // nkx
        j = m % nkx

        k1 = kx[j] * a1[0] + ky[i] * a1[1]
        k2 = kx[j] * a2[0] + ky[i] * a2[1]

        my_image[n] = fun(k1 * nk, k2 * nk)

        status.update()

    if broadcast or comm.rank == 0:
        image = np.empty((nky, nkx), dtype=mesh.dtype)
    else:
        image = None

    comm.Gatherv(my_image, (image, sizes))

    if broadcast:
        comm.Bcast(image)

    if return_k:
        return kx, ky, image
    else:
        return image

def double_plot(mesh, q, nq, qxmin=-0.8, qxmax=0.8, qymin=-0.8, qymax=0.8,
        resolution=500, interpolation=elphmod.bravais.linear_interpolation,
        angle=60, outside=0.0, outlines=False, broadcast=True):
    """Show f(q1, q2, k1, k2) on "Brillouin zone made of Brillouin zones"."""

    nQ, nk, nk = mesh.shape

    fun = dict()

    q = np.around(np.array(q) / (2 * np.pi) * nq).astype(int)

    for iq, (Q1, Q2) in enumerate(q):
        fun[Q1, Q2] = interpolation(mesh[iq], angle=angle)

    nqx = int(round(resolution * (qxmax - qxmin)))
    nqy = int(round(resolution * (qymax - qymin)))

    # (note that endpoint=False is combined with a retstep/2 shift)
    qx, dqx = np.linspace(qxmin, qxmax, nqx, endpoint=False, retstep=True)
    qy, dqy = np.linspace(qymin, qymax, nqy, endpoint=False, retstep=True)

    qx += dqx / 2
    qy += dqy / 2

    qy = qy[::-1]

    a1, a2 = elphmod.bravais.translations(180 - angle)

    sizes, bounds = elphmod.MPI.distribute(nqy * nqx, bounds=True)

    my_image = np.empty(sizes[comm.rank], dtype=mesh.dtype)

    status = elphmod.misc.StatusBar(sizes[comm.rank], title='double plot')

    for n, m in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        i = m // nqx
        j = m % nqx

        q1 = qx[j] * a1[0] + qy[i] * a1[1]
        q2 = qx[j] * a2[0] + qy[i] * a2[1]

        q1 *= nq
        q2 *= nq

        Q1, _ = divmod(q1, 1)
        Q2, _ = divmod(q2, 1)

        neighbors = [(Q1, Q2), (Q1, Q2 + 1), (Q1 + 1, Q2), (Q1 + 1, Q2 + 1)]

        nearest = min(neighbors, key=lambda q:
            elphmod.bravais.squared_distance(q[0] - q1, q[1] - q2, angle))

        if nearest in fun:
            my_image[n] = fun[nearest](q1 * nk, q2 * nk)
        else:
            my_image[n] = outside

        status.update()

    if broadcast or comm.rank == 0:
        image = np.empty((nqy, nqx), dtype=mesh.dtype)
    else:
        image = None

    comm.Gatherv(my_image, (image, sizes))

    if broadcast:
        comm.Bcast(image)

    if outlines:
        h = 1 / np.sqrt(3)
        a = 1 / 3

        b1, b2 = elphmod.bravais.reciprocals(a1, a2)

        miniBZ = []

        for q1, q2 in q:
            qx, qy = q1 * b1 + q2 * b2

            if angle == 90:
                miniBZ.append([
                    (qx + 0.5, qy - 0.5),
                    (qx + 0.5, qy + 0.5),
                    (qx - 0.5, qy + 0.5),
                    (qx - 0.5, qy - 0.5),
                    (qx + 0.5, qy - 0.5),
                ])
            else:
                miniBZ.append([
                    (qx + a * 2, qy),
                    (qx + a, qy + h),
                    (qx - a, qy + h),
                    (qx - a * 2, qy),
                    (qx - a, qy - h),
                    (qx + a, qy - h),
                    (qx + a * 2, qy),
                ])

        return image, np.array(miniBZ) / nq

    return image

def colorbar(image, left=0.02, bottom=0.02, width=0.03, height=0.30,
        minimum=None, maximum=None):
    """Add colorbar to image."""

    image_height, image_width = image.shape

    x1 = int(round(image_width * left))
    x2 = int(round(image_width * (left + width)))
    y1 = int(round(image_height * (1 - bottom)))
    y2 = int(round(image_height * (1 - bottom - height)))

    if minimum is None:
        minimum = np.nanmin(image)

    if maximum is None:
        maximum = np.nanmax(image)

    for y in range(min(y1, y2), max(y1, y2) + 1):
        image[y, x1:x2 + 1] = (
            minimum * (y2 - y) -
            maximum * (y1 - y)) / (y2 - y1)

def arrange(images, columns=None):
    if columns is None:
        columns = int(np.sqrt(len(images)))

    if columns < len(images):
        while len(images) % columns:
            columns += 1
    else:
        columns = len(images)

    rows = len(images) // columns

    images = [np.concatenate(images[columns * row:columns * (row + 1)], axis=1)
        for row in range(rows)]

    return np.concatenate(images, axis=0)

def toBZ(data=None, points=1000,
        interpolation=elphmod.bravais.linear_interpolation, angle=120, angle0=0,
        outside=0.0, return_k=False, return_only_k=False, even=False,
        broadcast=True):
    """Map data on uniform grid onto (wedge of) Brillouin zone.

    Parameters
    ----------
    angle : int
        Angle between first and second Bravais-lattice vector in degrees.
    angle0 : float
        Angle between x axis and first Bravais-lattice vector in degrees.
    """
    a1, a2 = elphmod.bravais.translations(angle, angle0)
    b1, b2 = elphmod.bravais.reciprocals(a1, a2)

    if angle == 60:
        a3 = a1 - a2
        b3 = b1 + b2
        M = 2 / 3

    elif angle == 90:
        a3 = a1
        b3 = b1
        M = 1 / 2

    elif angle == 120:
        a3 = a1 + a2
        b3 = b1 - b2
        M = 2 / 3

    kxmax = max(abs(a1[0]), abs(a2[0]), abs(a3[0])) * M
    kymax = max(abs(a1[1]), abs(a2[1]), abs(a3[1])) * M

    nkx = int(round(points * kxmax))
    nky = int(round(points * kymax))

    if even:
        nkx += nkx % 2
        nky += nky % 2

    # (note that endpoint=False is combined with a retstep/2 shift)
    kx, dkx = np.linspace(-kxmax, kxmax, nkx, endpoint=False, retstep=True)
    ky, dky = np.linspace(-kymax, kymax, nky, endpoint=False, retstep=True)

    kx += dkx / 2
    ky += dky / 2

    ky = ky[::-1]

    if return_only_k:
        return kxmax, kymax, kx, ky

    if data.ndim == 2:
        data = data[np.newaxis]

    ndata, nk, nk = data.shape

    fun = list(map(interpolation, data))

    sizes, bounds = elphmod.MPI.distribute(nky * nkx, bounds=True)

    my_image = np.empty(sizes[comm.rank],
        dtype=float if np.isrealobj(data) else complex)

    my_image[:] = outside

    angle0 *= np.pi / 180
    scale = ndata / (2 * np.pi)

    status = elphmod.misc.StatusBar(sizes[comm.rank], title='BZ plot')

    for n, m in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        i = m // nkx
        j = m % nkx

        k = np.array([kx[j], ky[i]])

        k1 = np.dot(k, a1)
        k2 = np.dot(k, a2)

        status.update()

        if abs(np.dot(k, b1)) > M: continue
        if abs(np.dot(k, b2)) > M: continue
        if abs(np.dot(k, b3)) > M: continue

        idata = np.floor((np.arctan2(ky[i], kx[j]) - angle0)
            * scale).astype(int) % ndata

        my_image[n] = fun[idata](k1 * nk, k2 * nk)

    if broadcast or comm.rank == 0:
        image = np.empty((nky, nkx), dtype=my_image.dtype)
    else:
        image = None

    comm.Gatherv(my_image, (image, sizes))

    if broadcast:
        comm.Bcast(image)

    if return_k:
        return kxmax, kymax, kx, ky, image
    else:
        return image

def rectify(image, width, height, lt, rt, lb, rb, *args, **kwargs):
    """Map skew image selection onto rectangular area.

    This function could turn a photograph of a document and its surroundings,
    taken at any unfortunate angle, into a rectangular image of the document
    content alone (similar to common scan applications).

    Parameters
    ----------
    image : ndarray
        2D array with image data.
    width : float
        Output width in units of original image width.
    height : float
        Output height in units of original image height.
    lt : tuple of float
        Upper-left corner of rectangle as original image coordinates.
    rt : tuple of float
        Upper-right corner of rectangle as original image coordinates.
    lb : tuple of float
        Lower-left corner of rectangle as original image coordinates.
    rb : tuple of float
        Lower-right corner of rectangle as original image coordinates.
    *args, **kwargs
        Arguments passed to linear-interpolation routine.

    Returns
    -------
    ndarray
        Rectangular view of image selection.
    """
    Ny, Nx = image.shape

    nx = int(round(Nx * width))
    ny = int(round(Ny * height))

    lt = np.array([Nx * lt[0], Ny * lt[1]])
    rt = np.array([Nx * rt[0], Ny * rt[1]])
    lb = np.array([Nx * lb[0], Ny * lb[1]])
    rb = np.array([Nx * rb[0], Ny * rb[1]])

    fun = elphmod.bravais.linear_interpolation(image, *args, **kwargs)

    image = np.empty((ny, nx))

    for iy in range(ny):
        l = (lb * iy + lt * (ny - iy)) / ny
        r = (rb * iy + rt * (ny - iy)) / ny

        for ix in range(nx):
            ixf, iyf = (r * ix + l * (nx - ix)) / nx

            image[iy, ix] = fun(Ny - iyf, ixf)

    return image

def color(data, cmap=None, minimum=None, maximum=None, comm=comm):
    """Colorize data using colormap from StoryLines (parallelized version).

    Parameters
    ----------
    data : list of list
        Data on two-dimensional mesh.
    cmap : function
        Colormap from StoryLines.
    minimum, maxmimum : float
        Data values corresponding to minimum and maximum of color scale.

    Returns
    -------
    list of list of list
        RGB image.
    """
    if minimum is None:
        minimum = np.nanmin(data)

    if maximum is None:
        maximum = np.nanmax(data)

    for n, x in enumerate(cmap.x):
        if type(x) is str:
            cmap.x[n] = (float(x) - minimum) / (maximum - minimum)

    data = (data - minimum) / (maximum - minimum)

    data = np.maximum(data, 0) # Avoid that data slightly exceeds [0, 1]
    data = np.minimum(data, 1) # because of numerical inaccuracies.

    shape = data.shape
    data = data.flatten()

    sizes, bounds = elphmod.MPI.distribute(data.size, bounds=True, comm=comm)

    my_image = np.empty((sizes[comm.rank], 3))

    status = elphmod.misc.StatusBar(sizes[comm.rank], title='color')

    for my_i, i in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        my_image[my_i] = cmap(data[i]).RGB()
        status.update()

    image = np.empty((*shape, 3))

    comm.Allgatherv(my_image, (image, sizes * 3))

    return image

def adjust_pixels(image, points, distances, width, height=None):
    """Even out image with block-wise equidistant columns.

    Parameters
    ----------
    image : ndarray
        Image with inconsistent horizontal (2nd axis) sampling.
    points : list of int
        Indices of columns where equidistant sampling changes.
    distances : list of float
        Desired cumulative distance of `points`.
    width : int
        Desired width of adjusted image.
    height : int, optional
        Desired height of adjusted image. Unchanged by default.

    Returns
    -------
    ndarray
        Adjusted image with overall equidistant columns.
    """
    distances = np.array(distances) - distances[0]

    new_points = np.round(distances / distances[-1] * (width - 1)).astype(int)

    if height is None:
        height = image.shape[0]

    new_image = np.empty((height, width, *image.shape[2:]))

    for i in range(1, len(points)):
        new_image[:, new_points[i - 1]:new_points[i] + 1] \
            = elphmod.bravais.resize(image[:, points[i - 1]:points[i] + 1],
                angle=90, periodic=False,
                shape=(height, new_points[i] - new_points[i - 1] + 1))

    return new_image

def compline(x, y, composition, center=True):
    """Plot composition along line."""

    nx = len(composition)
    composition = np.reshape(composition, (nx, -1))
    nc = composition.shape[1]

    lines = np.zeros((nc + 1, nx))

    for ic in range(nc):
        lines[ic + 1] = lines[ic] + composition[:, ic]

    if center:
        for ic in range(nc + 1):
            lines[ic] += y - lines[nc] / 2

    X = np.empty(2 * nx + 1)
    Y = np.empty((nc, 2 * nx + 1))

    X[:nx] = x
    X[nx:-1] = x[::-1]
    X[-1] = x[0]

    for ic in range(nc):
        Y[ic, :nx] = lines[ic]
        Y[ic, nx:-1] = lines[ic + 1, ::-1]
        Y[ic, -1] = lines[ic, 0]

    XY = np.empty((nc, 2, 2 * nx + 1))

    sgn = 1

    for ic in range(nc):
        XY[ic, 0] = X[::sgn]
        XY[ic, 1] = Y[ic, ::sgn]

        sgn *= -1

    return XY
