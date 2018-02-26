#/usr/bin/env python

import numpy as np
from scipy.misc import toimage

import bravais

def toBZ(data, points=1000, outside=0.0, a=0, b=360):
    """Map data on uniform grid onto (wedge of) Brillouin zone."""

    nq, nq = data.shape

    M =     bravais.U1[0] / 2
    K = 2 * bravais.U2[1] / 3

    nqx = int(round(points * M))
    nqy = int(round(points * K))

    qx = np.linspace(-M, M, nqx)
    qy = np.linspace(K, -K, nqy)

    image = np.empty((nqy, nqx))
    image[:] = outside

    U1 = bravais.U1 / np.sqrt(np.dot(bravais.U1, bravais.U1))
    U2 = bravais.U2 / np.sqrt(np.dot(bravais.U2, bravais.U2))
    U3 = U2 - U1

    a %= 360
    b %= 360

    fun = bravais.linear_interpolation(data)

    for i in range(len(qy)):
        for j in range(len(qx)):
            q = np.array([qx[j], qy[i]])

            q1 = np.dot(q, bravais.T1)
            q2 = np.dot(q, bravais.T2)

            if abs(np.dot(q, U1)) > M: continue
            if abs(np.dot(q, U2)) > M: continue
            if abs(np.dot(q, U3)) > M: continue

            x = 180 * np.arctan2(qy[i], qx[j]) / np.pi % 360

            if a <= x < b or b <= a <= x or x < b <= a:
                image[i, j] = fun(q1 * nq, q2 * nq)

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

def plot_pie_with_TeX(filename, image,
    title = 'Title',
    labels = range(1, 7),

    size           = 4.0, # distance between K and -K in plot (cm)
    margin         = 0.5, # lower margin (cm)
    title_spacing  = 0.2, # cm
    colorbar_width = 0.5, # cm

    lower = -2.0,
    upper = +2.0,

    ticks = np.linspace(-2.0, +2.0, 5),
    unit = 'Unit',

    positive = (241, 101, 34),
    negative = (54, 99, 173),
    ):
    """Create 'pie diagram' of different data on Brillouin zone."""


    with open(filename, 'w') as TeX:
        # write LaTeX header:

        x_dim = size + 2 * margin + 2.0
        y_dim = size + 3 * margin + title_spacing

        TeX.write(r'''\documentclass{{article}}

\usepackage[paperwidth={x_dim}cm, paperheight={y_dim}cm, margin=0cm]{{geometry}}
\usepackage[math]{{iwona}}
\usepackage{{tikz}}

\definecolor{{positive}}{{RGB}}{{{positive[0]}, {positive[1]}, {positive[2]}}}
\definecolor{{negative}}{{RGB}}{{{negative[0]}, {negative[1]}, {negative[2]}}}

\setlength\parindent{{0pt}}

\begin{{document}}'''.format(**locals()))

        # add frames and labels to Brillouin-zone plot:

        r = 0.5 * size
        R = r + margin

        y_title = R + title_spacing
        y_top   = R + title_spacing + margin

        angles = range(0, 360, 60)
        label_list = ','.join('%d/%s' % pair for pair in zip(angles, labels))

        TeX.write(r'''
    \begin{{tikzpicture}}[line join=round, line cap=round]
        \useasboundingbox (-{R}, -{R}) rectangle ({R}, {y_top});
        \node at (0, {y_title}) {{\large \bf \color{{negative}} {title}}};
        \node {{\includegraphics[height={size}cm]{{{image}}}}};
        \foreach \angle in {{ 30, 90, ..., 330 }}
            \draw [gray] (0, 0) -- (\angle:{r}cm) -- (\angle+60:{r}cm);
        \foreach \angle/\label in {{ {label_list} }}
            \node at (\angle:2cm) [rotate=\angle-90] {{\label}};
    \end{{tikzpicture}}%'''.format(**locals()))

        # print colorbar:

        x_unit = 0.5 * colorbar_width
        y_top = size + title_spacing + 2 * margin
        y_zero = -size * lower / (upper - lower)

        positions = [size * (tick - lower) / (upper - lower) for tick in ticks]
        tick_list = ','.join('%.3f/%+g' % pair
            for pair in zip(positions, ticks))

        TeX.write(r'''
    \begin{{tikzpicture}}
        \useasboundingbox (0, -{margin}) rectangle (2, {y_top});
        \shade [bottom color=negative, top color=white]
            (0, 0) rectangle ({colorbar_width}, {y_zero});
        \shade [bottom color=white, top color=positive]
            (0, {y_zero}) rectangle ({colorbar_width}, {size});
        \draw [gray] (0, 0) rectangle ({colorbar_width}, {size});
        \node [above] at ({x_unit}, {size}) {{{unit}}};
        \foreach \position/\label in {{ {tick_list} }}
            \node [right] at ({colorbar_width}, \position) {{$\label$}};
    \end{{tikzpicture}}%
\end{{document}}'''.format(**locals()))

