#!/usr/bin/env python3

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import copy
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

    def test_q2r(self):
        """Test Fourier interpolation of dynamical matrix."""

        for divide_mass in False, True:
            el, ph, elph, elel = elphmod.models.graphene.create(
                divide_mass=divide_mass)
            ph2 = copy.copy(ph)

            for irr in False, True:
                if irr:
                    nq = elphmod.models.graphene.nq[0]

                    q = np.array(sorted(elphmod.bravais.irreducibles(nq)),
                        dtype=float) * 2 * np.pi / nq
                else:
                    q = elphmod.models.graphene.q

                D = elphmod.dispersion.sample(ph.D, q)

                if irr:
                    elphmod.ph.q2r(ph2, D, q, nq, divide_mass=divide_mass)
                else:
                    elphmod.ph.q2r(ph2, D_full=D, divide_mass=divide_mass)

                ph2.standardize(eps=1e-10)

                self.assertTrue(np.allclose(ph.data, ph2.data))

if __name__ == '__main__':
    unittest.main()
