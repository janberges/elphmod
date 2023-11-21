#!/usr/bin/env python3

# Copyright (C) 2017-2023 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import numpy as np
import sys
import unittest

elphmod.misc.verbosity = 0

data = '../examples/data'

sys.path.append(data)

import graphene

class TestMD(unittest.TestCase):
    def test_dense_vs_sparse(self,
            N=2, kT=0.1, f=elphmod.occupations.fermi_dirac):
        """Verify that dense and sparse MD drivers yield identical results."""

        el = elphmod.el.Model('%s/graphene' % data, rydberg=True)
        ph = elphmod.ph.Model('%s/graphene.ifc' % data, divide_mass=False)
        elph = elphmod.elph.Model('%s/graphene.epmatwp' % data,
            '%s/graphene.wigner' % data, el, ph, divide_mass=False)

        ElPh = elph.supercell(N, N)

        driver_dense = elphmod.md.Driver(ElPh, kT, f, n=ElPh.el.size)

        driver_sparse = elphmod.md.Driver(elph, kT, f, n=elph.el.size,
            nk=(N, N), nq=(N, N), supercell=(N, N))

        driver_dense.random_displacements()
        driver_sparse.u[:] = driver_dense.u

        self.assertTrue(np.isclose(driver_dense.free_energy(show=False),
            driver_sparse.free_energy(show=False)))

        self.assertTrue(np.allclose(driver_dense.jacobian(show=False),
            driver_sparse.jacobian(show=False)))

if __name__ == '__main__':
    unittest.main()
