#!/usr/bin/env python3

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import copy
import elphmod.models.graphene
import numpy as np
import unittest

elphmod.misc.verbosity = 0

class TestElectronElectron(unittest.TestCase):
    def test_q2r(self):
        """Test Fourier interpolation of electron-electron interaction."""

        el, ph, elph, elel = elphmod.models.graphene.create()
        elel2 = copy.copy(elel)

        U = elphmod.dispersion.sample(elel.W, elphmod.models.graphene.q)

        elphmod.elel.q2r(elel2, U, ph.a, ph.r)

        elel2.standardize(eps=1e-10)

        self.assertTrue(np.allclose(elel.data, elel2.data))

if __name__ == '__main__':
    unittest.main()
