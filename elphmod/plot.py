#/usr/bin/env python

import numpy as np
from scipy.misc import toimage

from . import bravais, MPI
comm = MPI.comm

color1 = 0, 0, 255
color2 = 255, 0, 0

def plot(mesh, kxmin=-1.0, kxmax=1.0, kymin=-1.0, kymax=1.0, resolution=100,
        interpolation=bravais.linear_interpolation, angle=60):
    """Plot in cartesian reciprocal coordinates."""

    nk, nk = mesh.shape

    nkx = int(round(resolution * (kxmax - kxmin)))
    nky = int(round(resolution * (kymax - kymin)))

    kx, dkx = np.linspace(kxmin, kxmax, nkx, endpoint=False, retstep=True)
    ky, dky = np.linspace(kymin, kymax, nky, endpoint=False, retstep=True)

    ky = ky[::-1]

    kx += dkx / 2
    ky += dky / 2

    fun = interpolation(mesh, angle=angle)

    t1, t2 = bravais.translations(180 - angle)

    sizes, bounds = MPI.distribute(nky * nkx, bounds=True)

    my_image = np.empty(sizes[comm.rank])

    for n, m in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        i = m // nkx
        j = m % nkx

        k1 = kx[j] * t1[0] + ky[i] * t1[1]
        k2 = kx[j] * t2[0] + ky[i] * t2[1]

        my_image[n] = fun(k1 * nk, k2 * nk)

    image = np.empty((nky, nkx))

    comm.Gatherv(my_image, (image, sizes))

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

    qx, dqx = np.linspace(qxmin, qxmax, nqx, endpoint=False, retstep=True)
    qy, dqy = np.linspace(qymin, qymax, nqy, endpoint=False, retstep=True)

    qy = qy[::-1]

    qx += dqx / 2
    qy += dqy / 2

    t1, t2 = bravais.translations(180 - angle)

    sizes, bounds = MPI.distribute(nqy * nqx, bounds=True)

    my_image = np.empty(sizes[comm.rank])

    for n, m in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        i = m // nqx
        j = m % nqx

        q1 = qx[j] * t1[0] + qy[i] * t1[1]
        q2 = qx[j] * t2[0] + qy[i] * t2[1]

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

    image = np.empty((nqy, nqx))

    comm.Gatherv(my_image, (image, sizes))

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

    while len(images) % columns:
        columns += 1

    rows = len(images) // columns

    return \
        np.concatenate([
        np.concatenate(
            images[columns * row:columns * (row + 1)],
        axis=1) for row in range(rows)],
        axis=0)

def toBZ(data, points=1000, outside=0.0, angle=60):
    """Map data on uniform grid onto (wedge of) Brillouin zone."""

    if data.ndim == 2:
        data = data[np.newaxis]

    ndata, nk, nk = data.shape

    fun = list(map(bravais.linear_interpolation, data))

    t1, t2 = bravais.translations(180 - angle, angle0=angle - 90)
    u1, u2 = bravais.reciprocals(t1, t2)

    M = abs(u2[0])
    K = 2 * u2[1] / 3

    nkx = int(round(points * M))
    nky = int(round(points * K))

    kx = np.linspace(-M, M, nkx)
    ky = np.linspace(K, -K, nky)

    sizes, bounds = MPI.distribute(nky * nkx, bounds=True)

    my_image = np.empty(sizes[comm.rank])
    my_image[:] = outside

    u1 = u1 / np.sqrt(np.dot(u1, u1))
    u2 = u2 / np.sqrt(np.dot(u2, u2))

    if angle == 60:
        u3 = u1 - u2
    elif angle == 120:
        u3 = u1 + u2

    shift = 13.0 / 12.0

    for n, m in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        i = m // nkx
        j = m % nkx

        k = np.array([kx[j], ky[i]])

        k1 = np.dot(k, t1)
        k2 = np.dot(k, t2)

        if abs(np.dot(k, u1)) > M: continue
        if abs(np.dot(k, u2)) > M: continue
        if abs(np.dot(k, u3)) > M: continue

        idata = int((np.arctan2(ky[i], kx[j]) / (2 * np.pi) + shift)
            * ndata) % ndata

        my_image[n] = fun[idata](k1 * nk, k2 * nk)

    image = np.empty((nky, nkx))

    comm.Gatherv(my_image, (image, sizes))

    return image

def sign_color(data, negative=color1, positive=color2,
        minimum=None, maximum=None):
    """Transform gray-scale image to RGB, where zero is displayed as white."""

    lt0 = np.where(data < 0)
    gt0 = np.where(data > 0)

    image = data.copy()

    image[lt0] /= data.min() if minimum is None else minimum
    image[gt0] /= data.max() if maximum is None else maximum

    image = np.repeat(image[:, :, np.newaxis], 3, axis=-1)

    for c in range(3):
        image[:, :, c][lt0] *= 255 - negative[c]
        image[:, :, c][gt0] *= 255 - positive[c]

    return (255 - image).astype(int)

def HSV2RGB(H, S=1, V=1):
    """Transform hue, saturation, value to red, green, blue."""

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

def color(data, color1=(240, 1, 255), color2=(0, 1, 255), nancolor=(0, 0, 255),
        model='HSV', minimum=None, maximum=None):
    """Transform gray-scale image to RGB."""

    color1   = np.array(color1)
    color2   = np.array(color2)
    nancolor = np.array(nancolor)

    image = data.copy()

    image -= np.nanmin(image) if minimum is None else minimum
    image /= np.nanmax(image) if maximum is None else maximum

    new_image = np.empty(image.shape + (3,))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x = image[i, j]

            if np.isnan(x):
                new_image[i, j] = nancolor
            else:
                new_image[i, j] = (1 - x) * color1 + x * color2

            if model == 'HSV':
                new_image[i, j] = HSV2RGB(*new_image[i, j])

            elif model == 'PHI':
                new_image[i, j] = PHI2RGB(*new_image[i, j])

    return new_image

def save(filename, data):
    """Save image as 8-bit bitmap."""

    toimage(data, cmin=0, cmax=255).save(filename)

def label_pie_with_TeX(filename,
    imagename = None,

    width = 7.0, # total width in cm

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

    color1 = color1,
    color2 = color2,

    nCDW = 10,
    ):
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
        color1, color2)

    save('%s_colorbar.png' % stem, colorbar)

    if nCDW:
        t1, t2 = bravais.translations(120, angle0=-30)
        u1, u2 = bravais.reciprocals(t1, t2)

        A = sorted(set(n * n + n * m + m * m
          for n in range(13)
          for m in range(13)))[2:2 + nCDW]

        height_over_side = 0.5 * np.sqrt(3)

        kCDW = 1 / (np.sqrt(A) * height_over_side)

        indices = range(-12, 13)
        t = [(i, j) for i in indices for j in indices if i or j]
        T = [i * t1 + j * t2 for i, j in t]
        K = [bravais.rotate(t / t.dot(t), 90 * bravais.deg)
            / height_over_side for t in T]

        scaleCDW = GM / (0.5 * np.sqrt(np.dot(u1, u1)))

        KCDW = []

        for k in kCDW:
            KCDW.append([q * scaleCDW for q in K
                if abs(np.sqrt(q.dot(q)) - k) < 1e-10])

    X = locals()

    with open(filename, 'w') as TeX:
        # write embedding LaTeX document:

        TeX.write(r'''\documentclass{{article}}

\usepackage[paperwidth={width}cm, paperheight={height}cm, margin=0cm]{{geometry}}
\usepackage[math]{{iwona}}
\usepackage{{tikz}}

\setlength\parindent{{0pt}}

\begin{{document}}
  \input{{{stem}.tikz}}
\end{{document}}
'''.format(**X))

    with open('%s.tikz' % stem, 'w') as TeX:
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

def plot_pie_with_TeX(filename, data, points=1000,
        color1=color1, color2=color2, angle=60, **kwargs):
    """Create 'pie diagram' of different data on Brillouin zone."""

    data = np.array(data)

    image = toBZ(data, points=points, outside=np.nan, angle=angle)

    if comm.rank == 0:
        imagename = filename.rsplit('.', 1)[0] + '.png'
        save(imagename, color(image, color1, color2))

        label_pie_with_TeX(filename, imagename,
            color1=color1, lower=data.min(),
            color2=color2, upper=data.max(), **kwargs)

def compline(x, y, composition):
    """Plot composition along line."""

    nx, nc = composition.shape

    lines = np.zeros((nc + 1, nx))

    for ic in range(nc):
        lines[ic + 1] = lines[ic] + composition[:, ic]

    for ic in range(nc + 1):
        lines[ic] += y - lines[nc] / 2

    X = np.concatenate((x, x[::-1]))
    Y = np.empty((nc, len(X)))

    for ic in range(nc):
        Y[ic] = np.concatenate((lines[ic], lines[ic + 1, ::-1]))

    return X, Y

if __name__ == '__main__':
    import os

    os.system('mkdir -p plot_test')
    os.chdir('plot_test')

    label_pie_with_TeX('pie_plot.tex')

    os.system('pdflatex pie_plot')
    os.chdir('..')
