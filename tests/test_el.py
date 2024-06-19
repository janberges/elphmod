#!/usr/bin/env python3

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import elphmod.models.graphene
import numpy as np
import unittest

elphmod.misc.verbosity = 0

class TestElectron(unittest.TestCase):
    def test_electron_cell_transforms(self, nk=4, N=2):
        """Verify that cell transformations leave electron energies untoched."""

        el, ph, elph, elel = elphmod.models.graphene.create(rydberg=True,
            divide_mass=False)

        ref = np.sort(elphmod.dispersion.dispersion_full(el.H, nk), axis=None)

        el = el.supercell(N, N)
        e = elphmod.dispersion.dispersion_full(el.H, nk // N)
        self.assertTrue(np.allclose(np.sort(e, axis=None), ref))

        el = el.unit_cell()
        e = elphmod.dispersion.dispersion_full(el.H, nk)
        self.assertTrue(np.allclose(np.sort(e, axis=None), ref))

        el.shift_orbitals(0, (-1, 0, 0))
        e = elphmod.dispersion.dispersion_full(el.H, nk)
        self.assertTrue(np.allclose(np.sort(e, axis=None), ref))

        el.order_orbitals(1, 0)
        e = elphmod.dispersion.dispersion_full(el.H, nk)
        self.assertTrue(np.allclose(np.sort(e, axis=None), ref))

if __name__ == '__main__':
    unittest.main()
