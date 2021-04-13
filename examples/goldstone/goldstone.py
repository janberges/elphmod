#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.patches as pts
import matplotlib.pyplot as plt
import numpy as np

colors = ['dodgerblue', 'orange']
labels = ['$x, y$', '$z$']

e = elphmod.el.read_pwo('pw.out')[0][0]
nel = len(e)
e = e.reshape((1, 1, nel))

ph = dict()
d = dict()

for method in 'cdfpt', 'dfpt':
    ph[method] = elphmod.ph.Model(method + '.ifc')
    ph[method].data *= elphmod.misc.Ry ** 2
    nph = ph[method].size

    d[method] = elphmod.elph.read_xml_files(method + '.phsave/elph.%d.%d.xml',
        q=1, rep=nph, bands=range(nel), nbands=nel, nk=1, status=False)
    d[method] *= elphmod.misc.Ry ** 1.5

ph['dfpt_asr'] = elphmod.ph.Model('dfpt.ifc', apply_asr=True)
ph['dfpt_asr'].data *= elphmod.misc.Ry ** 2

ph['dfpt_rsr'] = elphmod.ph.Model('dfpt.ifc', apply_rsr=True)
ph['dfpt_rsr'].data *= elphmod.misc.Ry ** 2

d2 = np.einsum('qiklmn,qjklmn->qijklmn', d['cdfpt'].conj(), d['dfpt'])
d2 += np.einsum('qijklmn->qjiklmn', d2.conj())
d2 /= 2

q = np.array([[0, 0]])
Pi = elphmod.diagrams.phonon_self_energy(q, e, d2,
    occupations=elphmod.occupations.heaviside)
Pi = Pi.reshape((nph, nph))

if elphmod.MPI.comm.rank != 0:
    raise SystemExit

D = [
    ph['dfpt'].D(),
    ph['cdfpt'].D(),
    ph['cdfpt'].D() + Pi / ph['dfpt'].M[0],
    ph['dfpt_asr'].D(),
    ph['dfpt_rsr'].D(),
    ]

Shorten = 0.1
shorten = 0.01
width = 5.0

plt.axhline(0.0, color='gray', zorder=0)

for n in range(len(D)):
    X1 = n - 0.5 + Shorten
    X2 = n + 0.5 - Shorten

    w2, u = np.linalg.eigh(D[n])
    w = elphmod.ph.sgnsqrt(w2) * 1e3

    Z = (abs(u[2::3]) ** 2).sum(axis=0) > 0.5

    for group in elphmod.misc.group(w, eps=1.1 * width):
        N = len(group)

        for i, nu in enumerate(group):
            x1 = (i * X2 + (N - i) * X1) / N + shorten; i += 1
            x2 = (i * X2 + (N - i) * X1) / N - shorten

            plt.fill_between([x1, x2], w[nu] - 0.5 * width, w[nu] + 0.5 * width,
                linewidth=0.0, color=colors[int(Z[nu])])

plt.xticks(range(len(D)),
    ['DFPT', 'cDFPT', 'cDFPT+$\Pi$', 'DFPT+ASR', 'DFPT+RSR'])
plt.ylabel('phonon frequency (meV)')

plt.legend(handles=[pts.Patch(color=color, label=label)
    for color, label in zip(colors, labels)])

plt.show()
