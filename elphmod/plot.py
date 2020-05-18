#/usr/bin/env python

import numpy as np

import os

if 'DISPLAY' not in os.environ:
    # switch to non-GUI backend:

    import matplotlib
    matplotlib.use('agg')

import matplotlib.pyplot as plt

from . import bravais, MPI, misc
comm = MPI.comm

color1 = 0, 0, 255
color2 = 255, 0, 0

def plot(mesh, kxmin=-1.0, kxmax=1.0, kymin=-1.0, kymax=1.0, resolution=100,
        interpolation=bravais.linear_interpolation, angle=60, return_k=False):
    """Plot in cartesian reciprocal coordinates."""

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

    image = np.empty((nky, nkx) if comm.rank == 0 else (0, 0), dtype=mesh.dtype)

    comm.Gatherv(my_image, (image, sizes))

    if return_k:
        return kx, ky, image
    else:
        return image

def double_plot(mesh, q, nq, qxmin=-0.8, qxmax=0.8, qymin=-0.8, qymax=0.8,
        resolution=500, interpolation=bravais.linear_interpolation, angle=60,
        outside=0.0):
    """Show f(q1, q2, k1, k2) on "Brillouin zone made of Brillouin zones"."""

    nQ, nk, nk = mesh.shape

    fun = dict()

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

    image = np.empty((nqy, nqx), dtype=mesh.dtype)

    comm.Gatherv(my_image, (image, sizes))

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
        even=False):
    """Map data on uniform grid onto (wedge of) Brillouin zone.

    Parameters
    ----------
    angle : integer
        Angle between first and second Bravais-lattice vector in degrees.
    angle0 : float
        Angle between x axis and first Bravais-lattice vector in degrees.
    """
    a1, a2 = bravais.translations(angle, angle0)
    b1, b2 = bravais.reciprocals(a1, a2)

    if angle == 60:
        t3 = a1 - a2
        u3 = b1 + b2

    elif angle == 120:
        t3 = a1 + a2
        u3 = b1 - b2

    M = 2.0 / 3.0

    kxmax = max(abs(a1[0]), abs(a2[0]), abs(t3[0])) * M
    kymax = max(abs(a1[1]), abs(a2[1]), abs(t3[1])) * M

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
        if abs(np.dot(k, u3)) > M: continue

        idata = np.floor((np.arctan2(ky[i], kx[j]) - angle0)
            * scale).astype(int) % ndata

        my_image[n] = fun[idata](k1 * nk, k2 * nk)

    image = np.empty((nky, nkx) if comm.rank == 0 else (0, 0), dtype=data.dtype)

    comm.Gatherv(my_image, (image, sizes))

    if return_k:
        return kxmax, kymax, kx, ky, image
    else:
        return image

class Color(object):
    def __init__(self, A=0, B=0, C=0, model='RGB'):
        self.A = A
        self.B = B
        self.C = C
        self.model = model

    def __repr__(self):
        return '%s(%g, %g, %g)' % (self.model, self.A, self.B, self.C)

    def __add__(i, u):
        if i.model == u.model:
            if i.model == 'RGB':
                return Color(i.A + u.A, i.B + u.B, i.C + u.C)
            if i.model == 'HSV'\
            or i.model == 'PSV':
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

    def RGB(self):
        if self.model == 'HSV':
            return HSV2RGB(self.A, self.B, self.C)
        elif self.model == 'PSV':
            return PSV2RGB(self.A, self.B, self.C)
        else:
            return self.A, self.B, self.C

    def toRGB(self):
        return Color(*self.RGB())

def color_scheme(*args):
    default = Color(255, 255, 255)
    points = []

    for arg in args:
        if arg[0] is None:
            default = arg[1]
        else:
            points.append((arg[0], arg[1], arg[2] if len(arg) > 2 else None))

    values, colors, functions = zip(*points)

    def color(value):
        for n in range(len(points) - 1):
            if values[n] <= value <= values[n + 1]:
                weight = (value - values[n]) / (values[n + 1] - values[n])

                if functions[n] is not None:
                    weight = functions[n](weight)

                return (1 - weight) * colors[n] + weight * colors[n + 1]

        return default

    return color

def color_new(data, scheme, minimum=None, maximum=None):
    if minimum is None:
        minimum = np.nanmin(data)

    if maximum is None:
        maximum = np.nanmax(data)

    data = (data - minimum) / (maximum - minimum)

    image = np.empty(data.shape + (3,))

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            x = data[i, j]

            image[i, j] = scheme(data[i, j]).RGB()

    return image

def sign_color(data, negative=color1, neutral=(255, 255, 255), positive=color2,
        minimum=None, maximum=None):
    """Transform gray-scale image to RGB, where zero is displayed as white."""

    image = data.copy()

    image[np.where(np.isnan(image))] = 0

    lt0 = np.where(image <  0)
    ge0 = np.where(image >= 0)

    image[lt0] /= data.min() if minimum is None else minimum
    image[ge0] /= data.max() if maximum is None else maximum

    image = np.repeat(image[:, :, np.newaxis], 3, axis=-1)

    for c in range(3):
        image[:, :, c][lt0] = (negative[c] * image[:, :, c][lt0]
            + neutral[c] * (1 - image[:, :, c][lt0]))
        image[:, :, c][ge0] = (positive[c] * image[:, :, c][ge0]
            + neutral[c] * (1 - image[:, :, c][ge0]))

    return image.astype(int)

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

def PHI2RGB(alpha, beta, gamma):
    """Set color via phases of RGB channels (0 -> 0, pi -> 255, 2 pi -> 0)."""

    return 255 * (0.5 - 0.5 * np.cos(np.array([alpha, beta, gamma])))

def PSV2RGB(P=0, S=1, V=255):
    """Set color via phase, shift, and value."""

    return V * (0.5 - 0.5 * np.cos(P + S * np.array([0, 1, 2])))

def color(data, color1=(240, 1, 255), color2=(0, 1, 255), nancolor=(0, 0, 255),
        model='HSV', minimum=None, maximum=None, exponent=1, color3=None,
        scheme=None, **ignore):
    """Transform gray-scale image to RGB."""

    if scheme is not None:
        return color_new(data, minimum=minimum, maximum=maximum, scheme=scheme)

    color1   = np.array(color1)
    color2   = np.array(color2)
    nancolor = np.array(nancolor)

    if minimum is None:
        minimum = np.nanmin(data)

    if maximum is None:
        maximum = np.nanmax(data)

    if color3 is not None:
        color3 = np.array(color3)

        image = data.copy()

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if not np.isnan(image[i, j]):
                    if image[i, j] > 0:
                        image[i, j] /= maximum
                        image[i, j] **= exponent
                    elif image[i, j] < 0:
                        image[i, j] /= minimum
                        image[i, j] **= exponent
                        image[i, j] *= -1
    else:
        image = (data - minimum) / (maximum - minimum)
        image **= exponent

    new_image = np.empty(image.shape + (3,))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x = image[i, j]

            if np.isnan(x):
                new_image[i, j] = nancolor
            elif x < 0:
                new_image[i, j] = (1 + x) * color1 - x * color3
            else:
                new_image[i, j] = (1 - x) * color1 + x * color2

            if model == 'HSV':
                new_image[i, j] = HSV2RGB(*new_image[i, j])

            elif model == 'PHI':
                new_image[i, j] = PHI2RGB(*new_image[i, j])

    return new_image

def save(filename, data):
    """Save image as 8-bit bitmap."""

    plt.imsave(filename, data.astype(np.uint8))

def label_pie_with_TeX(filename,
    imagename = None,

    width = 7.0, # total width in cm

    preamble = r'\usepackage[math]{iwona}',

    # dimensions in arbitrary units:

    width_L   = 5.0, # width of part left of colorbar (Brillouin zone)
    width_R   = 1.0, # width of part right of colorbar (ticks)
    width_C   = 0.5, # width of colorbar
    spacing   = 0.5, # minimum spacing around Brillouin zone
    spacing_T = 0.7, # extra spacing for title on top

    title = None,
    label = None, # e.g. '(a)'
    labels = ['Label %d' % _ for _ in range(1, 7)],

    upper = +1.0,
    lower = -1.0,

    ticks = [-1.0, 0.0, 1.0],
    form  = lambda x: '$%g$' % x,
    unit  = 'Unit',

    nCDW = 10,

    standalone = True,

    **coloring):
    """Label 'pie diagram' of different data on Brillouin zone."""

    radius = 0.5 * width_L

    GK = radius - spacing      # Gamma to K
    GM = 0.5 * np.sqrt(3) * GK # Gamma to M
    KK = 2 * GK                # -K to K

    x_max = radius + width_C + width_R
    y_max = radius

    if title is not None:
        y_max += spacing_T

        y_title = radius + 0.4 * spacing_T

    x_unit  = radius + width_C * 0.5
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

    stem = filename.rsplit('.', 1)[0]

    colorbar = color(np.reshape(np.linspace(upper, lower, 300), (-1, 1)),
        **coloring)

    save('%s_colorbar.png' % stem, colorbar)

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

    with open(filename, 'w') as TeX:
        # write embedding LaTeX document:

        if standalone:
            TeX.write(r'''\documentclass{{article}}

\usepackage[paperwidth={width}cm, paperheight={height}cm, margin=0cm]{{geometry}}
\usepackage{{tikz}}
{preamble}

\setlength\parindent{{0pt}}

\begin{{document}}
'''.format(**X))

        # write ebmedded LaTeX code:

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

        if imagename is not None:
            TeX.write(r'''
  \node {{\includegraphics[height={KK}\unit]{{{imagename}}}}};'''.format(**X))

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
    pdf=False, **kwargs):
    """Create 'pie diagram' of different data on Brillouin zone."""

    data = np.array(data)

    image = toBZ(data, points=points, outside=np.nan, angle0=-30, angle=angle)

    if comm.rank == 0:
        imagename = filename.rsplit('.', 1)[0] + '.png'
        save(imagename, color(image, **kwargs))

        label_pie_with_TeX(filename, imagename,
            lower=data.min(), upper=data.max(), standalone=standalone, **kwargs)

        if standalone and pdf:
            os.system('pdflatex --interaction=batchmode ' + filename)

            for suffix in 'aux', 'log':
                os.system('rm %s' % filename.replace('tex', suffix))

def compline(x, y, composition, center=True):
    """Plot composition along line."""

    nx, nc = composition.shape

    lines = np.zeros((nc + 1, nx))

    for ic in range(nc):
        lines[ic + 1] = lines[ic] + composition[:, ic]

    if center:
        for ic in range(nc + 1):
            lines[ic] += y - lines[nc] / 2

    X = np.empty(     2 * nx + 1 )
    Y = np.empty((nc, 2 * nx + 1))

    X[  :nx] = x
    X[nx:-1] = x[::-1]
    X[   -1] = x[0]

    for ic in range(nc):
        Y[ic,   :nx] = lines[ic]
        Y[ic, nx:-1] = lines[ic + 1, ::-1]
        Y[ic,    -1] = lines[ic, 0]

    XY = np.empty((nc, 2, 2 * nx + 1))

    sgn = 1

    for ic in range(nc):
        XY[ic, 0] = X[    ::sgn]
        XY[ic, 1] = Y[ic, ::sgn]

        sgn *= -1

    return XY

if __name__ == '__main__':
    import os

    os.system('mkdir -p plot_test')
    os.chdir('plot_test')

    label_pie_with_TeX('pie_plot.tex')

    os.system('pdflatex pie_plot')
    os.chdir('..')
