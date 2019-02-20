#/usr/bin/env python

import elphmod

import numpy as np
import matplotlib.pyplot as plt

comm = elphmod.MPI.comm
info = elphmod.MPI.info

Ry2eV = 13.605693009
eV2cmm1 = 8065.54

data = 'NbSe2-cDFPT-LR'

info("Read and fix force constants and set up dynamical matrix..")

ph = elphmod.ph.Model('data/%s.ifc' % data, apply_asr=True)

info("Check module against Quantum ESPRESSO's 'matdyn.x'..")

q, x = elphmod.bravais.GMKG()

w2, e, order = elphmod.dispersion.dispersion(ph.D, q,
    vectors=True, order=True, broadcast=False)

w = elphmod.ph.sgnsqrt(w2) * Ry2eV * eV2cmm1

if comm.rank == 0:
    pol = elphmod.ph.polarization(e, q)

    colors = ['skyblue', 'dodgerblue', 'orange']

    ref = np.loadtxt('data/%s.disp.gp' % data)

    x0 = ref[:, 0] / ref[-1, 0] * x[-1]
    w0 = ref[:, 1:]

    for i in range(w.shape[1]):
        X, Y = elphmod.plot.compline(x, w[:, i], 3 * pol[:, i])

        for j in range(3):
            plt.fill(X, Y[j], color=colors[j], linewidth=0.0)

        plt.plot(x0, w0[:, i], 'ko')

    plt.show()

info("Calculate dispersion on whole Brillouin zone and sort bands..")

nq = 48

w2, order = elphmod.dispersion.dispersion_full(ph.D, nq, order=True)

w = elphmod.ph.sgnsqrt(w2) * Ry2eV * eV2cmm1

if comm.rank == 0:
    plt.plot(range(nq * nq), np.reshape(w, (nq * nq, ph.size)))
    plt.show()

info("Load and preprocess electron-phonon coupling..")

nqelph = 12

elph = np.empty((nqelph, nqelph, ph.size))

if comm.rank == 0:
    elph[:] = elphmod.elph.read('data/%s.elph' % data, nqelph, ph.size)

    step = nq // nqelph
    orderelph = order[::step, ::step]

    for n in range(nqelph):
        for m in range(nqelph):
            elph[n, m] = elph[n, m, orderelph[n, m]]

comm.Bcast(elph)

plots = [elphmod.plot.plot(elph[:, :, nu]) for nu in range(ph.size)]

if comm.rank == 0:
    plt.imshow(elphmod.plot.arrange(plots))
    plt.show()

g2 = np.empty_like(w)

if comm.rank == 0:
    scale = 1.0 / step

    for nu in range(ph.size):
        elphfun = elphmod.bravais.Fourier_interpolation(elph[:, :, nu])

        for n in range(nq):
            for m in range(nq):
                g2[n, m, nu] = elphfun(scale * n, scale * m)

    g2 /= 2 * w

comm.Bcast(g2)

info("Calculate DOS and a2F via 2D tetrahedron method..")

N = 300

W = np.linspace(w.min(), w.max(), N)

DOS = np.zeros(N)
a2F = np.zeros(N)

for nu in range(ph.size):
    DOS += elphmod.dos.hexDOS(w[:, :, nu])(W)
    a2F += elphmod.dos.hexa2F(w[:, :, nu], g2[:, :, nu])(W)

if comm.rank == 0:
    a2F *= DOS.max() / a2F.max()

    plt.fill_between(W, 0, DOS, facecolor='lightgray')
    plt.fill_between(W, 0, a2F, facecolor='blue', alpha=0.4)

    plt.show()
