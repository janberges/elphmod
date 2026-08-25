#!/usr/bin/env python3

# Copyright (C) 2017-2026 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import unittest

class TestMPI(unittest.TestCase):
    def test_shared_memory(self):
        """Verify that shared memory works as expected."""

        array = elphmod.MPI.SharedArray(1, dtype=int)

        if array.node.rank == array.node.size - 1:
            array[0] = 42

        array.node.Barrier()

        self.assertEqual(array[0], 42)

if __name__ == '__main__':
    unittest.main()
