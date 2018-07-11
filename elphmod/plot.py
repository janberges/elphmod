#/usr/bin/env python

import numpy as np
from scipy.misc import toimage

from . import bravais, MPI
comm = MPI.comm

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

    t1 = np.array([1.0, 0.0])
    t2 = bravais.rotate(t1, (180 - angle) * bravais.deg)

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

def toBZ(data, points=1000, outside=0.0):
    """Map data on uniform grid onto (wedge of) Brillouin zone."""

    if data.ndim == 2:
        data = data[np.newaxis]

    ndata, nk, nk = data.shape

    fun = list(map(bravais.linear_interpolation, data))

    M =     bravais.U1[0] / 2
    K = 2 * bravais.U2[1] / 3

    nkx = int(round(points * M))
    nky = int(round(points * K))

    kx = np.linspace(-M, M, nkx)
    ky = np.linspace(K, -K, nky)

    sizes, bounds = MPI.distribute(nky * nkx, bounds=True)

    my_image = np.empty(sizes[comm.rank])
    my_image[:] = outside

    U1 = bravais.U1 / np.sqrt(np.dot(bravais.U1, bravais.U1))
    U2 = bravais.U2 / np.sqrt(np.dot(bravais.U2, bravais.U2))
    U3 = U2 - U1

    shift = 13.0 / 12.0

    for n, m in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        i = m // nkx
        j = m % nkx

        k = np.array([kx[j], ky[i]])

        k1 = np.dot(k, bravais.T1)
        k2 = np.dot(k, bravais.T2)

        if abs(np.dot(k, U1)) > M: continue
        if abs(np.dot(k, U2)) > M: continue
        if abs(np.dot(k, U3)) > M: continue

        idata = int((np.arctan2(ky[i], kx[j]) / (2 * np.pi) + shift)
            * ndata) % ndata

        my_image[n] = fun[idata](k1 * nk, k2 * nk)

    image = np.empty((nky, nkx))

    comm.Gatherv(my_image, (image, sizes))

    return image

def color(data, positive=(241, 101, 34), negative=(54, 99, 173)):
    """Transform gray-scale image to RGB, where zero is displayed as white."""

    lt0 = np.where(data < 0)
    gt0 = np.where(data > 0)

    image = data.copy()

    image[lt0] /= data.min()
    image[gt0] /= data.max()

    image = np.repeat(image[:, :, np.newaxis], 3, axis=-1)

    for c in range(3):
        image[:, :, c][lt0] *= 255 - negative[c]
        image[:, :, c][gt0] *= 255 - positive[c]

    return (255 - image).astype(int)

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

    positive = (241, 101, 34),
    negative = (54, 99, 173),

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

    positive = ', '.join(map(str, positive))
    negative = ', '.join(map(str, negative))

    if nCDW:
        A = sorted(set(n * n + n * m + m * m
          for n in range(13)
          for m in range(13)))[2:2 + nCDW]

        height_over_side = 0.5 * np.sqrt(3)

        kCDW = 1 / (np.sqrt(A) * height_over_side)

        indices = range(-12, 13)
        t = [(i, j) for i in indices for j in indices if i or j]
        T = [i * bravais.T1 + j * bravais.T2 for i, j in t]
        K = [bravais.rotate(t / t.dot(t), 90 * bravais.deg)
            / height_over_side for t in T]

        scaleCDW = GM / (0.5 * np.sqrt(np.dot(bravais.U1, bravais.U1)))

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
  \input{{{filename}.in}}
\end{{document}}
'''.format(**X))

    with open('%s.in' % filename, 'w') as TeX:
        # write ebmedded LaTeX code:

        TeX.write(r'''\begingroup%
\let\unit\relax%
\newlength\unit%
\setlength\unit{{{scale}\linewidth}}%'''.format(**X))

        if upper > 0:
            TeX.write(r'''
\definecolor{{positive}}{{RGB}}{{{positive}}}%'''.format(**X))

        if lower < 0:
            TeX.write(r'''
\definecolor{{negative}}{{RGB}}{{{negative}}}%'''.format(**X))

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

        if lower < 0:
            TeX.write(r'''
  \shade [bottom color=negative, top color=white]
    ({radius}, -{GK}) rectangle ({x_ticks}, {y_zero});'''.format(**X))

        if upper > 0:
            TeX.write(r'''
  \shade [bottom color=white, top color=positive]
    ({radius}, {y_zero}) rectangle ({x_ticks}, {GK});'''.format(**X))

        TeX.write(r'''
  \draw [gray]
    ({radius}, -{GK}) rectangle ({x_ticks}, {GK});
  \node [above] at ({x_unit}, {GK}) {{{unit}}};
  \foreach \position/\label in {{
    {ticks}}}
    \node [right] at ({x_ticks}, \position) {{\label}};'''.format(**X))

        if nCDW:
            for k, scale, a in zip(KCDW, kCDW / kCDW.max(), A):
                positions = sep.join('%g/%g' % tuple(xy) for xy in k)
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

def plot_pie_with_TeX(filename, data,
    points = 1000,
    positive = (241, 101, 34),
    negative = (54, 99, 173),
    **kwargs):
    """Create 'pie diagram' of different data on Brillouin zone."""

    data = np.array(data)

    image = toBZ(data, points=points)

    if comm.rank == 0:
        imagename = filename.rsplit('.', 1)[0] + '.png'
        save(imagename, color(image, positive=positive, negative=negative))

        label_pie_with_TeX(filename, imagename,
            positive=positive, lower=data.min(),
            negative=negative, upper=data.max(), **kwargs)

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
