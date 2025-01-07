#!/usr/bin/env python3

# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod.models.chain
import elphmod.models.graphene
import elphmod.models.tas2
import numpy as np
import unittest

elphmod.misc.verbosity = 0

class TestModels(unittest.TestCase):
    def test_chain(self):
        """Check if chain hopping decreases with increasing bond length."""

        el, ph, elph = elphmod.models.chain.create(rydberg=True,
            divide_mass=False)

        R = (1, 0, 0)

        self.assertTrue(np.all(elph.gR(*R, *R)[0] / el.t(*R) < 0))

    def test_graphene(self):
        """Check if graphene hopping decreases with increasing bond length."""

        el, ph, elph, elel = elphmod.models.graphene.create(rydberg=True,
            divide_mass=False)

        R = (0, 1, 0)

        ratio = -elphmod.models.graphene.beta / elphmod.models.graphene.tau
        self.assertTrue(ratio < 0)

        self.assertTrue(np.allclose(elph.gR(*R, *R)[1], el.t(*R) * ratio))

    def test_tas2(self):
        """Check if TMDC hopping decreases with increasing bond length."""

        el, ph, elph = elphmod.models.tas2.create(rydberg=True,
            divide_mass=False)

        R = (1, 0, 0)

        ratio = -elphmod.models.tas2.beta / np.linalg.norm(elph.ph.a[0])
        self.assertTrue(ratio < 0)

        self.assertTrue(np.allclose(elph.gR(*R, *R)[0], el.t(*R) * ratio))

if __name__ == '__main__':
    unittest.main()
