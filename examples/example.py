#/usr/bin/env python

import elphmod

import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
comm = MPI.COMM_WORLD

Ry2eV = 13.605693009
eV2cmm1 = 8065.54

data = 'NbSe2-cDFPT-LR'

if comm.rank == 0:
    print("Read and fix force constants and set up dynamical matrix..")

    model = elphmod.phonons.read_flfrc('data/%s.ifc' % data)

    elphmod.phonons.asr(model[0])
else:
    model = None

model = comm.bcast(model)

D = elphmod.phonons.dynamical_matrix(comm, *model)

if comm.rank == 0:
    print("Check module against Quantum ESPRESSO's 'matdyn.x'..")

q, x = elphmod.bravais.GMKG()

w2, e, order = elphmod.dispersion.dispersion(comm, D, q,
    vectors=True, order=True, broadcast=False)

w = elphmod.phonons.sgnsqrt(w2) * Ry2eV * eV2cmm1

if comm.rank == 0:
    pol = elphmod.phonons.polarization(e, q)

    colors = ['skyblue', 'dodgerblue', 'orange']

    ref = np.loadtxt('data/%s.disp.gp' % data)

    x0 = ref[:, 0] / ref[-1, 0] * x[-1]
    w0 = ref[:, 1:]

    for i in range(w.shape[1]):
        X, Y = elphmod.plot.compline(x, w[:, i], 3 * pol[:, i])

        for j in range(3):
            plt.fill(X, Y[j], color=colors[j])

        plt.plot(x0, w0[:, i], 'ko')

    plt.show()

    print("Calculate dispersion on whole Brillouin zone and sort bands..")

nq = 48

w2, order = elphmod.dispersion.dispersion_full(comm, D, nq, order=True)

w = elphmod.phonons.sgnsqrt(w2) * Ry2eV * eV2cmm1

if comm.rank == 0:
    plt.plot(range(nq * nq), np.reshape(w, (nq * nq, D.size)))
    plt.show()

    print("Load and preprocess electron-phonon elphmod.coupling..")

    nqelph = 12

    elph = elphmod.coupling.complete(elphmod.coupling.read('data/%s.elph' % data),
        nqelph, D.size) * (1e-3 * eV2cmm1) ** 3

    step = nq // nqelph
    orderelph = order[::step, ::step]

    for n in range(nqelph):
        for m in range(nqelph):
            elph[n, m] = elph[n, m, orderelph[n, m]]

    plt.imshow(elphmod.coupling.plot(elph))
    plt.show()

g2 = np.empty_like(w)

if comm.rank == 0:
    scale = 1.0 / step

    for nu in range(D.size):
        elphfun = elphmod.bravais.Fourier_interpolation(elph[:, :, nu])

        for n in range(nq):
            for m in range(nq):
                g2[n, m, nu] = elphfun(scale * n, scale * m)

    g2 /= 2 * w

comm.Bcast(g2)

if comm.rank == 0:
    print("Calculate DOS and a2F via 2D tetrahedron method..")

N = 300

W = np.linspace(w.min(), w.max(), N)

DOS = np.zeros(N)
a2F = np.zeros(N)

for nu in range(D.size):
    DOS += elphmod.dos.hexDOS(w[:, :, nu], comm)(W)
    a2F += elphmod.dos.hexa2F(w[:, :, nu], g2[:, :, nu], comm)(W)

if comm.rank == 0:
    a2F *= DOS.max() / a2F.max()

    plt.fill_between(W, 0, DOS, facecolor='lightgray')
    plt.fill_between(W, 0, a2F, facecolor='blue', alpha=0.4)

    plt.show()
