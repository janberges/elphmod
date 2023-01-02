#!/usr/bin/env python3

# Copyright (C) 2017-2023 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt
import numpy as np

comm = elphmod.MPI.comm
info = elphmod.MPI.info

el = elphmod.el.Model('TaS2')
mu = elphmod.el.read_Fermi_level('pw.out')
ph = elphmod.ph.Model('ifc', apply_asr_simple=True)
elph = elphmod.elph.Model('work/TaS2.epmatwp', 'wigner.dat', el, ph,
    divide_mass=False) # with this argument, elph.g() returns <k+q|dV/du|k>

k, x, GMKG = elphmod.bravais.GMKG(corner_indices=True)

e, order = elphmod.dispersion.dispersion(el.H, k, order=True)

if comm.rank == 0:
    plt.plot(x, e - mu)
    plt.ylabel('Electron energy (eV)')
    plt.xlabel('Wave vector')
    plt.xticks(x[GMKG], 'GMKG')
    plt.show()

w2, order = elphmod.dispersion.dispersion(ph.D, k, order=True)

if comm.rank == 0:
    plt.plot(x, elphmod.ph.sgnsqrt(w2) * elphmod.misc.Ry * 1e3)
    plt.ylabel('Phonon energy (meV)')
    plt.xlabel('Wave vector')
    plt.xticks(x[GMKG], 'GMKG')
    plt.show()

n = 0 # electron band index
nu = 0 # phonon band index
k1, k2 = -np.pi, np.pi / 2 # k point
q1, q2 = 0.0, np.pi # q point

e = np.linalg.eigvalsh(el.H(k1=k2, k2=k2))
e -= mu
info('Electron energy e(k) = %g eV' % e[n])

w2 = np.linalg.eigvalsh(ph.D(q1=q1, q2=q2))
w = elphmod.ph.sgnsqrt(w2) * elphmod.misc.Ry
info('Phonon energy w(q) = %g eV' % w[nu])

d = elph.g(q1=q1, q2=q2, k1=k1, k2=k2, elbnd=True, phbnd=True)
d *= elphmod.misc.Ry / elphmod.misc.a0
info('Deformation potential <k+q|dV/du|k> = %g eV/AA' % abs(d[nu, n, n]))
