#!/usr/bin/env python3

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import numpy as np
import unittest

class TestWigner(unittest.TestCase):
    def _test_2d(self, angle=120, nk=12):
        """Verify that 2D and 3D (general) implementations identical results."""

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

    def test_2d_60(self):
        self._test_2d(60)

    def test_2d_90(self):
        self._test_2d(90)

    def test_2d_120(self):
        self._test_2d(120)

if __name__ == '__main__':
    unittest.main()
