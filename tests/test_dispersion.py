#!/usr/bin/env python3

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import elphmod.models.graphene
import numpy as np
import unittest

elphmod.misc.verbosity = 0

class TestDispersion(unittest.TestCase):
    def test_dispersion_full(self, nk=12):
        """Validate mapping of dispersion from irreducible wedge to full BZ."""

        el, ph, elph, elel = elphmod.models.graphene.create(rydberg=True,
            divide_mass=False)

        e_sym = elphmod.dispersion.dispersion_full(el.H, nk)
        e_nosym = elphmod.dispersion.dispersion_full_nosym(el.H, nk)

        self.assertTrue(np.allclose(e_sym, e_nosym))

if __name__ == '__main__':
    unittest.main()
