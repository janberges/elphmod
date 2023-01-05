#!/usr/bin/env python3

# Copyright (C) 2017-2023 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

__all__ = ['__drivers__', 'Dummy_driver']

import sys
sys.path.append('..')

import data.graphene
import elphmod
import subprocess

el = elphmod.el.Model('../data/graphene', rydberg=True)
ph = elphmod.ph.Model('../data/graphene.ifc', divide_mass=False)
elph = elphmod.elph.Model('../data/graphene.epmatwp', '../data/graphene.wigner',
    el, ph, divide_mass=False)

driver = elphmod.md.Driver(elph, kT=0.02, f=elphmod.occupations.fermi_dirac,
    n=elph.el.size, supercell=[(6, 0, 0), (3, 6, 0)])

driver.random_displacements(amplitude=0.1)

driver.to_xyz('init.xyz')

subprocess.Popen(['i-pi', 'input.xml'])

driver.plot(interactive=True)

def Dummy_driver(*ignore):
    return driver

__drivers__ = dict(dummy=Dummy_driver)
