#/usr/bin/env python

import elphmod
import matplotlib.pyplot as plt
import numpy as np
import os

comm = elphmod.MPI.comm
info = elphmod.MPI.info

mu = -0.1665
kT = 0.01
nk = 800

step = 20
Nk = nk // step

q = [0.5, 0.0]

logkT = np.linspace(-8, 2, 100)

info('Calculate electron dispersion')

el = elphmod.el.Model('data/NbSe2_hr.dat')

if os.path.exists('el.npy'):
    if comm.rank == 0:
        ekk = np.load('el.npy')
    else:
        ekk = np.empty((nk, nk))

    comm.Bcast(ekk)

else:
    ekk = elphmod.dispersion.dispersion_full(el.H, nk)[:, :, 0] - mu

    if comm.rank == 0:
        np.save('el.npy', ekk)

Ekk = ekk[::step, ::step].copy()

info('Calculate DDI on full q BZ')

kT = 0.01

delta_kk = elphmod.occupations.fermi_dirac.delta(ekk / kT) / kT

q_irr = np.array(sorted(elphmod.bravais.irreducibles(Nk)))

DDI_irr = np.empty((len(q_irr), 4))

progress = elphmod.misc.StatusBar(len(q_irr))

for iq, (q1, q2) in enumerate(step * q_irr):
    ekq = np.roll(np.roll(ekk, shift=q1, axis=0), shift=q2, axis=1)
    Ekq = ekq[::step, ::step].copy()

    intersections, DDI = elphmod.dos.double_delta(Ekk, Ekq)(0)

    delta_kq = elphmod.occupations.fermi_dirac.delta(ekq / kT) / kT

    DDI_irr[iq, 0] = len(intersections)
    DDI_irr[iq, 1] = DDI.sum()
    DDI_irr[iq, 2] = np.average(delta_kk * delta_kq)

    progress.update()

DDI_irr[..., 3] = np.absolute(DDI_irr[..., 1] - DDI_irr[..., 2])

DDI = np.empty((Nk, Nk, 4))

for iq, (q1, q2) in enumerate(q_irr):
    for Q1, Q2 in elphmod.bravais.images(q1, q2, Nk):
        DDI[Q1, Q2] = DDI_irr[iq]

images = [elphmod.plot.toBZ(DDI[..., n], points=501) for n in range(4)]

if comm.rank == 0:
    fig, ax = plt.subplots(1, 4)

    for n in range(4):
        ax[n].imshow(images[n], cmap='inferno', vmax=5 if n else None)

    plt.show()

info('Calculate density of states (DOS) with tetrahedron')

DOS_tetra = elphmod.dos.hexDOS(Ekk)(0)

info('Calculate double-delta integral (DDI) with tetrahedron')

ekq = np.roll(np.roll(ekk,
    shift=int(round(q[0] * nk)), axis=0),
    shift=int(round(q[1] * nk)), axis=1)

Ekq = ekq[::step, ::step].copy()

intersections, DDI_tetra = elphmod.dos.double_delta(Ekk, Ekq)(0)

DDI_tetra = DDI_tetra.sum()

info('Calculate DOS and DDI for different smearings')

kT = 10 ** logkT

sizes, bounds = elphmod.MPI.distribute(len(kT), bounds=True)

my_DOS_smear = np.empty(sizes[comm.rank])
my_DDI_smear = np.empty(sizes[comm.rank])

for my_n, n in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
    print('kT = %g eV' % kT[n])

    delta_kk = elphmod.occupations.fermi_dirac.delta(ekk / kT[n]) / kT[n]
    delta_kq = elphmod.occupations.fermi_dirac.delta(ekq / kT[n]) / kT[n]

    my_DOS_smear[my_n] = np.average(delta_kk)
    my_DDI_smear[my_n] = np.average(delta_kk * delta_kq)

DOS_smear = np.empty(len(kT))
DDI_smear = np.empty(len(kT))

comm.Gatherv(my_DOS_smear, (DOS_smear, sizes))
comm.Gatherv(my_DDI_smear, (DDI_smear, sizes))

info('Plot Fermi surfaces')

contours_kk = []

plot = dict(kxmin=-0.8, kxmax=0.8, kymin=-0.7, kymax=0.7,
    return_k=True, resolution=101)

kx, ky, BZ_kk = elphmod.plot.plot(ekk, **plot)
kx, ky, BZ_kq = elphmod.plot.plot(ekq, **plot)

if comm.rank == 0:
    a1, a2 = elphmod.bravais.translations()
    b1, b2 = elphmod.bravais.reciprocals(a1, a2)

    outline = list(zip(*[(k1 * b1 + k2 * b2) / 3 for k1, k2
        in [(2, -1), (1, 1), (-1, 2), (-2, 1), (-1, -1), (1, -2), (2, -1)]]))

    intersections = list(zip(*[(K1 * b1 + K2 * b2) / Nk
        for k1, k2 in intersections
        for K1, K2 in elphmod.bravais.to_Voronoi(k1, k2, Nk)]))

    plt.contour(kx, ky, BZ_kk, colors='k', levels=[0.0])
    plt.contour(kx, ky, BZ_kq, colors='k', levels=[0.0])
    plt.plot(*outline, 'b')
    plt.plot(*intersections, 'ob')
    plt.axis('equal')
    plt.show()

info('Plot results')

if comm.rank == 0:
    plt.xlabel('log($k T$/eV)')
    plt.ylabel('DOS/eV$^{-1}$, DDI/eV$^{-2}$')
    plt.axhline(y=DOS_tetra, color='r', label='DOS (tetra.)')
    plt.axhline(y=DDI_tetra, color='b', label='DDI (tetra.)')
    plt.plot(logkT, DOS_smear, 'or', label='DOS (smear.)')
    plt.plot(logkT, DDI_smear, 'ob', label='DDI (smear.)')
    plt.legend()
    plt.show()
