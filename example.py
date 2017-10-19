#/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import bravais
import coupling
import dos
import phonons

Ry2eV = 13.605693009
eV2cmm1 = 8065.54

print("Read and fix force constants and set up dynamical matrix..")

phid, amass, at, tau = phonons.read_flfrc('data/NbSe2-cDFPT-SR.ifc')

phonons.asr(phid)

D = phonons.dynamical_matrix(phid, amass, at, tau)

bands = D().shape[0]

print("Check module against Quantum ESPRESSO's 'matdyn.x'..")

path, x = bravais.GMKG()

w = np.empty((len(path), bands))

for n, q in enumerate(path):
    w[n] = phonons.frequencies(D(*q))

w *= Ry2eV * eV2cmm1

ref = np.loadtxt('data/NbSe2-cDFPT-SR.disp.gp')

x0 = ref[:, 0] / ref[-1, 0] * x[-1]
w0 = ref[:, 1:]

for i in range(w.shape[1]):
    plt.plot(x,  w [:, i], 'k' )
    plt.plot(x0, w0[:, i], 'ko')

plt.show()

print("Calculate dispersion on whole Brillouin zone and sort bands..")

nq = 48

w, order = phonons.dispersion(D, nq)
w *= Ry2eV * eV2cmm1

plt.plot(range(nq * nq), np.reshape(w, (nq * nq, bands)))
plt.show()

print("Load and preprocess electron-phonon coupling..")

nqelph = 12

elph = coupling.complete(coupling.read('data/NbSe2-cDFPT-LR.elph'),
    nqelph, bands) * (1e-3 * eV2cmm1) ** 3

step = nq // nqelph
orderelph = order[::step, ::step]

for n in range(nqelph):
    for m in range(nqelph):
        elph[n, m] = elph[n, m, orderelph[n, m]]

plt.imshow(coupling.plot(elph))
plt.show()

g2 = np.empty_like(w)

scale = 1.0 / step

for n in range(nq):
    for m in range(nq):
        for nu in range(bands):
            g2[n, m, nu] = bravais.interpolate(elph[:, :, nu],
                scale * n, scale * m)

g2 /= 2 * w

print("Calculate DOS and a2F via 2D tetrahedron method..")

N = 300

W = np.linspace(w.min(), w.max(), N)

DOS = np.zeros(N)
a2F = np.zeros(N)

for nu in range(9):
    DOS += dos.hexDOS(w[:, :, nu])(W)
    a2F += dos.hexa2F(w[:, :, nu], g2[:, :, nu])(W)

a2F *= DOS.max() / a2F.max()

plt.fill_between(W, 0, DOS, facecolor='lightgray')
plt.fill_between(W, 0, a2F, facecolor='blue', alpha=0.4)

plt.show()
