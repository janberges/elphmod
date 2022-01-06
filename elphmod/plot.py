#!/usr/bin/env python3

# Copyright (C) 2017-2022 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import numpy as np

from . import bravais, MPI, misc
comm = MPI.comm

def choose_backend():
    """Switch to non-GUI Matplotlib backend if necessary."""

    import os
    import matplotlib

    if 'DISPLAY' not in os.environ:
        matplotlib.use('agg')

def plot(mesh, kxmin=-1.0, kxmax=1.0, kymin=-1.0, kymax=1.0, resolution=100,
        interpolation=bravais.linear_interpolation, angle=60, return_k=False,
        broadcast=True):
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

    a1, a2 = bravais.translations(180 - angle)

    sizes, bounds = MPI.distribute(nky * nkx, bounds=True)

    my_image = np.empty(sizes[comm.rank], dtype=mesh.dtype)

    status = misc.StatusBar(sizes[comm.rank], title='plot')

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
        resolution=500, interpolation=bravais.linear_interpolation, angle=60,
        outside=0.0, broadcast=True):
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

    a1, a2 = bravais.translations(180 - angle)

    sizes, bounds = MPI.distribute(nqy * nqx, bounds=True)

    my_image = np.empty(sizes[comm.rank], dtype=mesh.dtype)

    status = misc.StatusBar(sizes[comm.rank], title='double plot')

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
            bravais.squared_distance(q[0] - q1, q[1] - q2, angle))

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

    return image

def double_plot_tex(texfile, imgfile, q, nq, angle=60,
        qxmin=-0.8, qxmax=0.8, qymin=-0.8, qymax=0.8, scale=10.0):
    """Draw outlines of mini Brillouin zones.

    See Also
    --------
    double_plot
    """
    q = np.array(q, dtype=float) / nq

    h = 1.0 / (nq * np.sqrt(3))
    a = 1.0 / (nq * 3)

    a1, a2 = bravais.translations(180 - angle)
    b1, b2 = bravais.reciprocals(a1, a2)

    with open(texfile, 'w') as TeX:
        TeX.write(r'''\documentclass{{article}}
\usepackage[paperwidth={width}cm, paperheight={height}cm, margin=0cm]{{geometry}}
\usepackage{{tikz}}
\setlength\parindent{{0pt}}
\begin{{document}}
\begin{{tikzpicture}}[x={scale}cm, y={scale}cm]
    \useasboundingbox ({qxmin}, {qymin}) rectangle ({qxmax}, {qymax});
    \node [anchor=south west, inner sep=0, outer sep=0] at ({qxmin}, {qymin})
        {{\includegraphics[width={width}cm, height={height}cm]{{{imgfile}}}}};
'''.format(width=scale * (qxmax - qxmin), height=scale * (qymax - qymin),
            scale=scale, qxmin=qxmin, qxmax=qxmax, qymin=qymin, qymax=qymax,
            imgfile=imgfile))

        for q1, q2 in q:
            qx, qy = q1 * b1 + q2 * b2

            points = [
                (qx + a * 2, qy),
                (qx + a, qy + h),
                (qx - a, qy + h),
                (qx - a * 2, qy),
                (qx - a, qy - h),
                (qx + a, qy - h),
                ]

            points = ' -- '.join('(%.4f, %.4f)' % point for point in points)

            TeX.write(r'''\draw [white, line width=4pt] {points} -- cycle;
'''.format(points=points))

        TeX.write(r'''\end{tikzpicture}%
\vspace*{-1cm}%
\end{document}''')

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

    return \
        np.concatenate([
        np.concatenate(
            images[columns * row:columns * (row + 1)],
        axis=1) for row in range(rows)],
        axis=0)

def toBZ(data=None, points=1000, interpolation=bravais.linear_interpolation,
        angle=120, angle0=0, outside=0.0, return_k=False, return_only_k=False,
        even=False, broadcast=True):
    """Map data on uniform grid onto (wedge of) Brillouin zone.

    Parameters
    ----------
    angle : int
        Angle between first and second Bravais-lattice vector in degrees.
    angle0 : float
        Angle between x axis and first Bravais-lattice vector in degrees.
    """
    a1, a2 = bravais.translations(angle, angle0)
    b1, b2 = bravais.reciprocals(a1, a2)

    if angle == 60:
        a3 = a1 - a2
        b3 = b1 + b2
        M = 2.0 / 3.0

    elif angle == 90:
        a3 = a1
        b3 = b1
        M = 1.0 / 2.0

    elif angle == 120:
        a3 = a1 + a2
        b3 = b1 - b2
        M = 2.0 / 3.0

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

    sizes, bounds = MPI.distribute(nky * nkx, bounds=True)

    my_image = np.empty(sizes[comm.rank], dtype=data.dtype)
    my_image[:] = outside

    angle0 *= np.pi / 180
    scale = ndata / (2 * np.pi)

    status = misc.StatusBar(sizes[comm.rank], title='BZ plot')

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
        image = np.empty((nky, nkx), dtype=data.dtype)
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

    fun = bravais.linear_interpolation(image, *args, **kwargs)

    image = np.empty((ny, nx))

    for iy in range(ny):
        l = (lb * iy + lt * (ny - iy)) / ny
        r = (rb * iy + rt * (ny - iy)) / ny

        for ix in range(nx):
            ixf, iyf = (r * ix + l * (nx - ix)) / nx

            image[iy, ix] = fun(Ny - iyf, ixf)

    return image

class Color(object):
    """Representation of a color.

    Colors can be defined using different models:

        RGB (red, green, blue):
            The three RGB channels take values from 0 to 255.
        HSV (hue, saturation, value):
            The hue is the color angle in degrees from 0 to 360.
            Values of 0, 120, and 240 correspond to red, green, and blue.
            The saturation takes values from -1 to 1.
            A value of 0 is white; negative values yield complementary colors.
            The value is the RGB amplitude from 0 to 255.
            A value of 0 is black.
        PSV (phase, shift, value):
            Superposition of shifted RGB "waves".
            The phase and the phase shift from R to G (G to B) are in radians.
            The value is the amplitude of the wave from 0 to 255.

    Colors can be mixed:

    .. code-block:: python

        red = Color(255, 0, 0, 'RGB')
        green = Color(120, 1, 255, 'HSV')
        yellow = (red + green) / 2

    Here, colors of different models are converted to RGB first.
    """
    def __init__(self, A, B, C, model='RGB'):
        self.A = A
        self.B = B
        self.C = C
        self.model = model

    context = None

    def __str__(self):
        if self.context is None:
            return '%s(%g, %g, %g)' % (self.model, self.A, self.B, self.C)

        RGB = tuple(np.around(self.RGB()).astype(int))

        if self.context == 'TeX':
            return '{rgb,255:red,%d;green,%d;blue,%d}' % RGB

        if self.context == 'HTML':
            return '#%02x%02x%02x' % RGB

    def __repr__(self):
        return 'Color(%g, %g, %g, %s)' % (self.A, self.B, self.C, self.model)

    def __add__(i, u):
        if i.model == u.model:
            if i.model == 'RGB':
                return Color(i.A + u.A, i.B + u.B, i.C + u.C)
            if i.model == 'HSV'\
            or i.model == 'PSV':
                if i.C == u.C == 0:
                    return Color((i.A + u.A) / 2, (i.B + u.B) / 2, 0, i.model)

                return Color(
                    (i.A * i.C + u.A * u.C) / (i.C + u.C),
                    (i.B * i.C + u.B * u.C) / (i.C + u.C),
                    i.C + u.C, i.model)
        else:
            return i.toRGB() + u.toRGB()

    def __mul__(i, u):
        if i.model == 'RGB':
            return Color(i.A * u, i.B * u, i.C * u)
        if i.model == 'HSV'\
        or i.model == 'PSV':
            return Color(i.A, i.B, i.C * u, i.model)

    __rmul__ = __mul__

    def __sub__(i, u):
        return i + (-1 * u)

    def __truediv__(i, u):
        return i * u ** -1

    __div__ = __truediv__

    def RGB(self):
        if self.model == 'HSV':
            return HSV2RGB(self.A, self.B, self.C)
        elif self.model == 'PSV':
            return PSV2RGB(self.A, self.B, self.C)
        else:
            return self.A, self.B, self.C

    def toRGB(self):
        return Color(*self.RGB())

def colormap(*args):
    """Map interval [0, 1] to colors.

    Colors can be defined for an arbitrary number of points in the interval. In
    between, colors are interpolated linearly. The color for the point ``None``
    is used for NaNs and beyond the outermost points where colors have been
    defined.

    Examples:

    .. code-block:: python

        bluebrown = colormap( # PRB 101, 155107 (2020)
            (0, Color(0.0, 1, 255, 'PSV'), np.sqrt),
            (1, Color(5.5, 1, 255, 'PSV')),
            (None, Color(255, 255, 255, 'RGB')),
            )

        AFMhot = colormap( # Gnuplot
            (0.00, Color(  0,   0,   0)),
            (0.25, Color(128,   0,   0)),
            (0.50, Color(255, 128,   0)),
            (0.75, Color(255, 255, 128)),
            (1.00, Color(255, 255, 255)),
            )
    """
    default = Color(255, 255, 255)
    points = []

    for arg in args:
        if arg[0] is None:
            default = arg[1]
        else:
            points.append((arg[0], arg[1], arg[2] if len(arg) > 2 else None))

    def color(x):
        for n in range(len(color.x) - 1):
            if color.x[n] <= x <= color.x[n + 1]:
                weight = (x - color.x[n]) / (color.x[n + 1] - color.x[n])

                if color.f[n] is not None:
                    weight = color.f[n](weight)

                return (1 - weight) * color.c[n] + weight * color.c[n + 1]

        return default

    color.x, color.c, color.f = tuple(map(list, zip(*points)))

    return color

def color(data, cmap=None, minimum=None, maximum=None, comm=comm):
    if minimum is None:
        minimum = np.nanmin(data)

    if maximum is None:
        maximum = np.nanmax(data)

    if cmap is None:
        cmap = colormap((0, Color(255, 255, 255)), (1, Color(0, 0, 0)))

    for n, x in enumerate(cmap.x):
        if type(x) is str:
            cmap.x[n] = (float(x) - minimum) / (maximum - minimum)

    data = (data - minimum) / (maximum - minimum)

    data = np.maximum(data, 0) # Avoid that data slightly exceeds [0, 1]
    data = np.minimum(data, 1) # because of numerical inaccuracies.

    shape = data.shape
    data = data.flatten()

    sizes, bounds = MPI.distribute(data.size, bounds=True, comm=comm)

    my_image = np.empty((sizes[comm.rank], 3))

    status = misc.StatusBar(sizes[comm.rank], title='color')

    for my_i, i in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        my_image[my_i] = cmap(data[i]).RGB()
        status.update()

    image = np.empty(shape + (3,))

    comm.Allgatherv(my_image, (image, sizes * 3))

    return image

def HSV2RGB(H, S=1, V=255):
    """Transform hue, saturation, value to red, green, blue."""

    if S < 0:
        S = -S
        H += 180

    H %= 360

    h = np.floor(H / 60)
    f = H / 60 - h

    p = V * (1 - S)
    q = V * (1 - S * f)
    t = V * (1 - S * (1 - f))

    if h == 0: return V, t, p
    if h == 1: return q, V, p
    if h == 2: return p, V, t
    if h == 3: return p, q, V
    if h == 4: return t, p, V
    if h == 5: return V, p, q

def RGB2HSV(R, G, B):
    """Transform red, green, blue to hue, saturation, value."""

    V = max(R, G, B)
    extent = V - min(R, G, B)

    if R == G == B:
        H = 0
    elif V == R:
        H = 60 * ((G - B) / extent)
    elif V == G:
        H = 60 * ((B - R) / extent + 2)
    elif V == B:
        H = 60 * ((R - G) / extent + 4)

    S = extent / V if V else 0

    return H, S, V

def PSV2RGB(P, S=1, V=255):
    """Set color via phase, shift, and value."""

    return tuple(V * (0.5 - 0.5 * np.cos(P + S * np.array([0, 1, 2]))))

def save(filename, image):
    """Save grayscale, RGB, or RGBA image as 8-bit PNG.

    Specified at https://www.w3.org/TR/PNG/.
    Inspired by Blender thumbnailer code.

    Parameters
    ----------
    filename : str
        Name of PNG file to be written.
    image : ndarray
        8-bit image data of shape (height, width, colors), where colors may be
        1 (grayscale), 3 (RGB), or 4 (RGBA).
    """
    import zlib, struct

    image[np.where(image < 0)] = 0
    image[np.where(image >= 256)] = 255

    height, width, colors = image.shape
    color = {1: 0, 3: 2, 4: 6}[colors]

    lines = np.empty((height, 1 + width * colors), dtype=np.uint8)
    lines[:, 0] = 0 # https://www.w3.org/TR/PNG/#4Concepts.EncodingFiltering
    lines[:, 1:] = image.reshape((height, -1))

    with open(filename, 'wb') as png:
        png.write(b'\x89PNG\r\n\x1a\n')

        def chunk(name, data):
            png.write(struct.pack('!I', len(data)))
            png.write(name)
            png.write(data)
            png.write(struct.pack('!I', zlib.crc32(name + data) & 0xffffffff))

        chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, color, 0, 0, 0))
        chunk(b'IDAT',
            zlib.compress(struct.pack('%dB' % lines.size, *lines.flat), 9))
        chunk(b'IEND', b'')

def load(filename):
    """Load grayscale, RGB, or RGBA image from 8-bit PNG.

    Parameters
    ----------
    filename : str
        Name of PNG file to be written.

    Returns
    -------
    ndarray
        8-bit image data of shape (height, width, colors), where colors may be
        1 (grayscale), 3 (RGB), or 4 (RGBA).
    """
    import zlib, struct

    with open(filename, 'rb') as png:
        png.read(8)

        while True:
            size, = struct.unpack('!I', png.read(4))
            name = png.read(4)
            data = png.read(size)
            csum = png.read(4)

            if struct.pack('!I', zlib.crc32(name + data) & 0xffffffff) != csum:
                print("Chunk '%s' corrupted!" % name)

            if name == b'IHDR':
                width, height, _, color, _, _, _ = struct.unpack('!2I5B', data)
                colors = {0: 1, 2: 3, 6: 4}[color]

            elif name == b'IDAT':
                data = zlib.decompress(data)
                lines = struct.unpack('%dB' % len(data), data)
                lines = np.reshape(lines, (height, 1 + width * colors))
                lines = lines.astype(np.uint8)
                image = lines[:, 1:].reshape((height, width, colors))

            if name == b'IEND':
                break

        return image

def label_pie_with_TeX(stem,
    width=7.0, # total width in cm

    preamble=r'\usepackage[math]{iwona}',

    # dimensions in arbitrary units:

    width_L=5.0, # width of part left of colorbar (Brillouin zone)
    width_R=1.0, # width of part right of colorbar (ticks)
    width_C=0.5, # width of colorbar
    spacing=0.5, # minimum spacing around Brillouin zone
    spacing_T=0.7, # extra spacing for title on top

    title=None,
    label=None, # e.g. '(a)'
    labels=['Label %d' % _ for _ in range(1, 7)],

    upper=+1.0,
    lower=-1.0,

    ticks=[-1.0, 0.0, 1.0],
    form=lambda x: '$%g$' % x,
    unit='Unit',

    nCDW=10,

    standalone=True,
    ):
    """Label 'pie diagram' of different data on Brillouin zone."""

    radius = 0.5 * width_L

    GK = radius - spacing # Gamma to K
    GM = 0.5 * np.sqrt(3) * GK # Gamma to M
    KK = 2 * GK # -K to K

    x_max = radius + width_C + width_R
    y_max = radius

    if title is not None:
        y_max += spacing_T

        y_title = radius + 0.4 * spacing_T

    x_unit = radius + width_C * 0.5
    x_ticks = radius + width_C

    def transform(y):
        return GK * (2 * (y - lower) / (upper - lower) - 1)

    y_zero = transform(0)

    sep = ',%\n    '
    ticks = sep.join('%g/{%s}' % (transform(_), form(_)) for _ in ticks)
    labels = sep.join('%d/%s' % _ for _ in zip(range(0, 360, 60), labels))

    x_dim = radius + x_max
    y_dim = radius + y_max

    height = width * y_dim / x_dim

    scale = 1 / x_dim

    if nCDW:
        a1, a2 = bravais.translations(120, angle0=-30)
        b1, b2 = bravais.reciprocals(a1, a2)

        A = sorted(set(n * n + n * m + m * m
          for n in range(13)
          for m in range(13)))[2:2 + nCDW]

        height_over_side = 0.5 * np.sqrt(3)

        kCDW = 1 / (np.sqrt(A) * height_over_side)

        indices = range(-12, 13)
        t = [(i, j) for i in indices for j in indices if i or j]
        T = [i * a1 + j * a2 for i, j in t]
        K = [bravais.rotate(t / t.dot(t), 90 * bravais.deg)
            / height_over_side for t in T]

        scaleCDW = GM / (0.5 * np.sqrt(np.dot(b1, b1)))

        KCDW = []

        for k in kCDW:
            KCDW.append([q * scaleCDW for q in K
                if abs(np.sqrt(q.dot(q)) - k) < 1e-10])

    X = locals()

    with open('%s.tex' % stem, 'w') as TeX:
        # write embedding LaTeX document:

        if standalone:
            TeX.write(r'''\documentclass{{article}}

\usepackage[paperwidth={width}cm, paperheight={height}cm, margin=0cm]{{geometry}}
\usepackage{{tikz}}
{preamble}

\setlength\parindent{{0pt}}

\begin{{document}}
'''.format(**X))

        # write embedded LaTeX code:

        TeX.write(r'''\begingroup%
\let\unit\relax%
\newlength\unit%
\setlength\unit{{{scale}\linewidth}}%'''.format(**X))

        # add frames and labels to Brillouin-zone plot:

        TeX.write(r'''
\begin{{tikzpicture}}[x=\unit, y=\unit]
  \useasboundingbox
    (-{radius}, -{radius}) rectangle ({x_max}, {y_max});'''.format(**X))

        if title is not None:
            TeX.write(r'''
  \node at (0, {y_title}) {{\large \bf {title}}};'''.format(**X))

        if label is not None:
            TeX.write(r'''
  \node [below right] at (-{radius}, {radius}) {{{label}}};'''.format(**X))

        TeX.write(r'''
  \node {{\includegraphics[height={KK}\unit]{{{stem}.png}}}};'''.format(**X))

        TeX.write(r'''
  \foreach \angle in {{ 30, 90, ..., 330 }}
    \draw [gray, line join=round, line cap=round]
      (0, 0) -- (\angle:{GK}) -- (\angle+60:{GK});
  \foreach \angle/\label in {{
    {labels}}}
    \node [above, rotate=\angle-90] at (\angle:{GM}) {{\label}};'''.format(**X))

        # print colorbar:

        TeX.write(r'''
  \node [inner sep=0, outer sep=0] at ({x_unit}, 0)
     {{\includegraphics[width={width_C}\unit, height={KK}\unit]
     {{{stem}_colorbar.png}}}};'''.format(**X))

        TeX.write(r'''
  \draw [gray]
    ({radius}, -{GK}) rectangle ({x_ticks}, {GK});
  \node [above] at ({x_unit}, {GK}) {{{unit}}};
  \foreach \position/\label in {{
    {ticks}}}
    \node [right] at ({x_ticks}, \position) {{\label}};'''.format(**X))

        if nCDW:
            for k, scale, a in zip(KCDW, kCDW / kCDW.max(), A):
                positions = sep.join('%.3f/%.3f' % tuple(xy) for xy in k)
                TeX.write(r'''
  \foreach \x/\y in {{
    {positions}}}
    \node [circle, inner sep=0.3pt, draw=gray, fill=white] at (\x, \y)
      {{\tiny \scalebox{{{scale}}}{{{a}}}}};'''.format(positions=positions,
            scale=scale, a=a))

        TeX.write(r'''
\end{tikzpicture}%
\endgroup%
''')

        if standalone:
            TeX.write(r'''\end{document}
''')

def plot_pie_with_TeX(filename, data, points=1000, angle=60, standalone=True,
    pdf=False, colormap=None, **kwargs):
    """Create 'pie diagram' of different data on Brillouin zone."""

    data = np.array(data)

    image = toBZ(data, points=points, outside=np.nan, angle0=-30, angle=angle)
    image = color(image, colormap)

    colorbar = np.reshape(np.linspace(1, 0, 300), (-1, 1))
    colorbar = color(colorbar, colormap)

    if comm.rank == 0:
        stem = filename.rsplit('.', 1)[0]
        save('%s.png' % stem, image)
        save('%s_colorbar.png' % stem, colorbar)

        label_pie_with_TeX(stem,
            lower=data.min(), upper=data.max(), standalone=standalone, **kwargs)

        if standalone and pdf:
            import subprocess

            subprocess.call(['pdflatex', '--interaction=batchmode', stem])

            for suffix in 'aux', 'log':
                os.remove('%s.%s' % (stem, suffix))

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
