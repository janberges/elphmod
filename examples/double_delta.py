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

ekq = np.roll(np.roll(ekk,
    shift=int(round(q[0] * nk)), axis=0),
    shift=int(round(q[1] * nk)), axis=1)

info('Calculate density of states (DOS) with tetrahedron')

DOS_tetra = elphmod.dos.hexDOS(ekk)(0)

info('Calculate double-delta integral (DDI) with tetrahedron')

intersections, DDI_tetra = elphmod.dos.double_delta(ekk, ekq)(0)

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
