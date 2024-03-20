#!/usr/bin/env python3

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import elphmod.models.tas2
import scipy.optimize

elphmod.models.tas2.create('data/TaS2')

el = elphmod.el.Model('data/TaS2', rydberg=True)
ph = elphmod.ph.Model('data/TaS2.ifc', divide_mass=False)
elph = elphmod.elph.Model('data/TaS2.epmatwp', 'data/TaS2.wigner',
    el, ph, divide_mass=False)

driver = elphmod.md.Driver(elph, nk=(12, 12), nq=(2, 2), supercell=(9, 9),
    kT=0.02, f=elphmod.occupations.marzari_vanderbilt, n=1.0)

driver.kT = 0.005
driver.f = elphmod.occupations.fermi_dirac

driver.random_displacements()

driver.plot(interactive=True)

scipy.optimize.minimize(driver.free_energy, driver.u, jac=driver.jacobian,
    method='BFGS', tol=1e-8)

driver.plot(interactive=False)
