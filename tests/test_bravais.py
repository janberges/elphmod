#!/usr/bin/env python3

# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import numpy as np
import unittest

class TestBravais(unittest.TestCase):
    def _test_wigner_2d(self, angle=120, nk=12):
        """Verify that 2D and general code yield same WS data."""

        at = np.eye(3)
        at[:2, :2] = elphmod.bravais.translations(angle)
        tau = np.zeros((1, 3))

        irvec, ndegen, wslen = elphmod.bravais.wigner(nk, nk, 1, at, tau)
        order = sorted(range(len(irvec)), key=lambda n: tuple(irvec[n]))

        irvec_2d, ndegen_2d, wslen_2d = elphmod.bravais.wigner_2d(nk, angle)
        order_2d = sorted(range(len(irvec_2d)), key=lambda n: irvec_2d[n])

        self.assertTrue(np.array_equal(irvec[order, :2],
            np.array(irvec_2d)[order_2d]))

        self.assertTrue(np.array_equal(ndegen[order, 0, 0],
            np.array(ndegen_2d)[order_2d]))

        self.assertTrue(np.allclose(wslen[order, 0, 0],
            np.array(wslen_2d)[order_2d]))

    def test_wigner_2d_60(self):
        """Verify that 2D (60 degrees) and general code yield same WS data."""

        self._test_wigner_2d(60)

    def test_wigner_2d_90(self):
        """Verify that 2D (90 degrees) and general code yield same WS data."""

        self._test_wigner_2d(90)

    def test_wigner_2d_120(self):
        """Verify that 2D (120 degrees) and general code yield same WS data."""

        self._test_wigner_2d(120)

    def test_images(self, k1=7, k2=3, nk=12):
        """Compare different functions to obtain equivalent k points."""

        for ibrav, angle in (4, 60), (6, 90):
            img1 = sorted(elphmod.bravais.images_ibrav(k1, k2, 1,
                nk, nk, 1, ibrav))
            img2 = sorted(elphmod.bravais.images(k1, k2, nk, angle))

            img2 = [(k1, k2, 0) for k1, k2 in img2]

            self.assertTrue(img1 == img2)

    def test_irreducibles(self, nk=12):
        """Compare different functions to obtain irreducible k points."""

        for ibrav, angle in (4, 60), (6, 90):
            irr1 = sorted(elphmod.bravais.irreducibles_ibrav(nk, nk, 1, ibrav))
            irr2 = sorted(elphmod.bravais.irreducibles(nk, angle))

            irr2 = [(k1, k2, 0) for k1, k2 in irr2]

            self.assertTrue(irr1 == irr2)

if __name__ == '__main__':
    unittest.main()
