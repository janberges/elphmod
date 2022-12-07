#!/usr/bin/env python3

# Copyright (C) 2017-2022 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt
import model
import scipy.optimize

comm = elphmod.MPI.comm

N = 6
nk = 6
nq = 6

kT0 = 2.0
kT = 0.02

el = elphmod.el.Model('data/graphene', rydberg=True)
ph = elphmod.ph.Model('data/graphene.ifc', divide_mass=False)
elph = elphmod.elph.Model('data/graphene.epmatwp', 'data/graphene.wigner',
    el, ph, divide_mass=False)

elph.data *= 1.5 # otherwise the system is stable

elph = elph.supercell(N, N)

driver = elphmod.md.Driver(elph, kT=kT0, f=elphmod.occupations.fermi_dirac,
    n=elph.el.size, nk=(nk // N,) * 2, nq=(nq // N,) * 2)

driver.kT = kT

driver.random_displacements(amplitude=0.05)

driver.plot()

scipy.optimize.minimize(driver.free_energy, driver.u, jac=driver.jacobian,
    method='BFGS', tol=1e-8)

driver.plot()

ph = driver.phonons()

path = 'GMKG'
q, x, corners = elphmod.bravais.path(path, ibrav=4, N=150)

w2 = elphmod.dispersion.dispersion(ph.D, q)

if comm.rank == 0:
    w = elphmod.ph.sgnsqrt(w2) * elphmod.misc.Ry * 1e3

    plt.plot(x, w, 'k')
    plt.ylabel('Phonon energy (meV)')
    plt.xlabel('Wave vector')
    plt.xticks(x[corners], path)
    plt.show()
