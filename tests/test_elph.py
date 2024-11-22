#!/usr/bin/env python3

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import copy
import elphmod.models.graphene
import numpy as np
import unittest

elphmod.misc.verbosity = 0

class TestElectronPhonon(unittest.TestCase):
    def test_q2r(self):
        """Test Fourier interpolation of electron-phonon coupling."""

        for divide_mass in False, True:
            el, ph, elph, elel = elphmod.models.graphene.create(
                divide_mass=divide_mass)
            elph2 = copy.copy(elph)

            g = elph.sample(elphmod.models.graphene.q.reshape((-1, 3)),
                elphmod.models.graphene.nk)

            elphmod.elph.q2r(elph2, elphmod.models.graphene.nq,
                elphmod.models.graphene.nk, g, ph.r, divide_mass)

            elph2.standardize(eps=1e-10)

            self.assertTrue(np.allclose(elph.data, elph2.data))

if __name__ == '__main__':
    unittest.main()
