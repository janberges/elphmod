#!/usr/bin/env python3

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import elphmod.models.graphene
import ipi._driver.driver
import subprocess
import time

elphmod.models.graphene.create('../data/graphene')

el = elphmod.el.Model('../data/graphene', rydberg=True)
ph = elphmod.ph.Model('../data/graphene.ifc', divide_mass=False)
elph = elphmod.elph.Model('../data/graphene.epmatwp', '../data/graphene.wigner',
    el, ph, divide_mass=False)

driver = elphmod.md.Driver(elph, kT=0.02, f=elphmod.occupations.fermi_dirac,
    n=elph.el.size, supercell=[(6, 0, 0), (3, 6, 0)])

driver.random_displacements(amplitude=0.1)

driver.to_xyz('init.xyz')

subprocess.Popen(['i-pi', 'input.xml'])

time.sleep(2) # wait for i-PI to load and create a socket

driver.plot(interactive=True)

ipi._driver.driver.run_driver(unix=True, address='localhost', driver=driver)
