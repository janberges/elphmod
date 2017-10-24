#/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import bravais
import coupling
import dos
import phonons

from mpi4py import MPI
comm = MPI.COMM_WORLD

Ry2eV = 13.605693009
eV2cmm1 = 8065.54

data = 'NbSe2-cDFPT-LR'

if comm.rank == 0:
    print("Read and fix force constants and set up dynamical matrix..")

    model = phonons.read_flfrc('data/%s.ifc' % data)

    phonons.asr(model[0])
else:
    model = None

model = comm.bcast(model)

D = phonons.dynamical_matrix(comm, *model)

bands = D().shape[0]

sizes = np.empty(comm.size, dtype=int)

if comm.rank == 0:
    print("Check module against Quantum ESPRESSO's 'matdyn.x'..")

    q, x = bravais.GMKG()
    w = np.empty((len(q), bands))

    sizes[:] = len(q) // comm.size
    sizes[:len(q) % comm.size] += 1
else:
    q = w = None

comm.Bcast(sizes)

my_q = np.empty((sizes[comm.rank], 2))
my_w = np.empty((sizes[comm.rank], bands))

comm.Scatterv((q, 2 * sizes), my_q)

for n, (q1, q2) in enumerate(my_q):
    my_w[n] = phonons.frequencies(D(q1, q2))

comm.Gatherv(my_w, (w, bands * sizes))

if comm.rank == 0:
    w *= Ry2eV * eV2cmm1

    ref = np.loadtxt('data/%s.disp.gp' % data)

    x0 = ref[:, 0] / ref[-1, 0] * x[-1]
    w0 = ref[:, 1:]

    for i in range(w.shape[1]):
        plt.plot(x,  w [:, i], 'k' )
        plt.plot(x0, w0[:, i], 'ko')

    plt.show()

if comm.rank == 0:
    print("Calculate dispersion on whole Brillouin zone and sort bands..")

nq = 48

w, order = phonons.dispersion(comm, D, nq)
w *= Ry2eV * eV2cmm1

if comm.rank == 0:
    plt.plot(range(nq * nq), np.reshape(w, (nq * nq, bands)))
    plt.show()

if comm.rank == 0:
    print("Load and preprocess electron-phonon coupling..")

    nqelph = 12

    elph = coupling.complete(coupling.read('data/%s.elph' % data),
        nqelph, bands) * (1e-3 * eV2cmm1) ** 3

    step = nq // nqelph
    orderelph = order[::step, ::step]

    for n in range(nqelph):
        for m in range(nqelph):
            elph[n, m] = elph[n, m, orderelph[n, m]]

    plt.imshow(coupling.plot(elph))
    plt.show()

g2 = np.empty_like(w)

if comm.rank == 0:
    scale = 1.0 / step

    for n in range(nq):
        for m in range(nq):
            for nu in range(bands):
                g2[n, m, nu] = bravais.interpolate(elph[:, :, nu],
                    scale * n, scale * m)

    g2 /= 2 * w

comm.Bcast(g2)

if comm.rank == 0:
    print("Calculate DOS and a2F via 2D tetrahedron method..")

N = 300

W = np.linspace(w.min(), w.max(), N)

DOS = np.zeros(N)
a2F = np.zeros(N)

for nu in range(bands):
    DOS += dos.hexDOS(w[:, :, nu], comm)(W)
    a2F += dos.hexa2F(w[:, :, nu], g2[:, :, nu], comm)(W)

if comm.rank == 0:
    a2F *= DOS.max() / a2F.max()

    plt.fill_between(W, 0, DOS, facecolor='lightgray')
    plt.fill_between(W, 0, a2F, facecolor='blue', alpha=0.4)

    plt.show()
