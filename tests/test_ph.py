#!/usr/bin/env python3

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import elphmod.models.graphene
import numpy as np
import unittest

elphmod.misc.verbosity = 0

class TestPhonon(unittest.TestCase):
    def test_phonon_cell_transforms(self, nq=4, N=2):
        """Verify that cell transformations leave phonon energies untoched."""

        el, ph, elph, elel = elphmod.models.graphene.create(rydberg=True,
            divide_mass=False)

        ref = np.sort(elphmod.dispersion.dispersion_full(ph.D, nq), axis=None)

        ph = ph.supercell(N, N)
        w2 = elphmod.dispersion.dispersion_full(ph.D, nq // N)
        self.assertTrue(np.allclose(np.sort(w2, axis=None), ref))

        ph = ph.unit_cell()
        w2 = elphmod.dispersion.dispersion_full(ph.D, nq)
        self.assertTrue(np.allclose(np.sort(w2, axis=None), ref))

        ph.shift_atoms(0, (-1, 0, 0))
        w2 = elphmod.dispersion.dispersion_full(ph.D, nq)
        self.assertTrue(np.allclose(np.sort(w2, axis=None), ref))

        ph.order_atoms(1, 0)
        w2 = elphmod.dispersion.dispersion_full(ph.D, nq)
        self.assertTrue(np.allclose(np.sort(w2, axis=None), ref))

if __name__ == '__main__':
    unittest.main()
