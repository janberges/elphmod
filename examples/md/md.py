#!/usr/bin/env python3

# Copyright (C) 2017-2026 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod.models.tas2
import subprocess
import time

try:
    import ipi_driver
except ModuleNotFoundError:
    import ipi._driver.driver as ipi_driver

el, ph, elph = elphmod.models.tas2.create(rydberg=True, divide_mass=False)

driver = elphmod.md.Driver(elph, kT=0.005, f='fd', n=1.0,
    nk=(12, 12), nq=(2, 2), supercell=(9, 9), kT0=0.02, f0='mv')

driver.random_displacements(amplitude=0.05, reproducible=True)

driver.to_xyz('init.xyz')

subprocess.Popen(['i-pi', 'input.xml'])

time.sleep(2) # wait for i-PI to load and create a socket

driver.plot(interactive=True)

ipi_driver.run_driver(unix=True, address='localhost', driver=driver)

driver.plot(interactive=False)
driver.plot(filename='md.png')
