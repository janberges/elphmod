#!/usr/bin/env python3

# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod.models.chain
import scipy.optimize

sparse = False # use sparse matrices to be able to simulate large cells?

N = 6
nk = 210
nq = 210

kT = 2e-4

elphmod.models.chain.create('data/chain')

el = elphmod.el.Model('data/chain', rydberg=True)
ph = elphmod.ph.Model('data/chain.ifc', divide_mass=False)
elph = elphmod.elph.Model('data/chain.epmatwp', 'data/chain.wigner',
    el, ph, divide_mass=False)

if not sparse:
    elph = elph.supercell(N)

    nk //= N
    nq //= N

driver = elphmod.md.Driver(elph, kT=kT, f=elphmod.occupations.fermi_dirac,
    nk=(nk,), nq=(nq,), supercell=(N,) if sparse else None, n=elph.el.size,
    unscreen=False, basis=[[0]])

driver.random_displacements(amplitude=0.05)

driver.plot(interactive=True, scale=20.0)

scipy.optimize.minimize(driver.free_energy, driver.u, jac=driver.jacobian,
    method='BFGS', options=dict(gtol=1e-8, norm=float('inf')))

driver.plot(interactive=False)
driver.plot(filename='ssh.png')
