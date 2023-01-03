#!/usr/bin/env python3

# Copyright (C) 2017-2023 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import data.chain
import elphmod
import scipy.optimize

sparse = False # use sparse matrices to be able to simulate large cells?

N = 6
nk = 210
nq = 210

kT = 2e-4

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
    unscreen=False)

driver.random_displacements(amplitude=0.05)

driver.plot(interactive=True, scale=20.0)

scipy.optimize.minimize(driver.free_energy, driver.u, jac=driver.jacobian,
    method='BFGS', tol=1e-8)

driver.plot(interactive=False)
