#!/usr/bin/env python3

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import unittest

class TestMisc(unittest.TestCase):
    def test_split(self):
        """Test factorizing expression with separators and brackets."""

        self.assertEqual(['dz2', 'dxz', 'dyz', 'dx2-y2', 'dxy'],
            list(elphmod.misc.split('d{z2, {x,y}z, x2-y2, xy}')))

if __name__ == '__main__':
    unittest.main()
