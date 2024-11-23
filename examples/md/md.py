#!/usr/bin/env python3

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod.models.tas2
import ipi._driver.driver
import subprocess
import time

el, ph, elph = elphmod.models.tas2.create(rydberg=True, divide_mass=False)

driver = elphmod.md.Driver(elph, nk=(12, 12), nq=(2, 2), supercell=(9, 9),
    kT=0.02, f=elphmod.occupations.marzari_vanderbilt, n=1.0)

driver.kT = 0.005
driver.f = elphmod.occupations.fermi_dirac

driver.random_displacements()

driver.to_xyz('init.xyz')

subprocess.Popen(['i-pi', 'input.xml'])

time.sleep(2) # wait for i-PI to load and create a socket

driver.plot(interactive=True)

ipi._driver.driver.run_driver(unix=True, address='localhost', driver=driver)

driver.plot(interactive=False)
