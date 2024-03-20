#!/usr/bin/env python3

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import elphmod.models.graphene
import numpy as np
import unittest

elphmod.misc.verbosity = 0

class TestMD(unittest.TestCase):
    def test_dense_vs_sparse(self,
            N=2, kT=0.1, f=elphmod.occupations.fermi_dirac):
        """Verify that dense and sparse MD drivers yield identical results."""

        elphmod.models.graphene.create('graphene')

        el = elphmod.el.Model('graphene', rydberg=True)
        ph = elphmod.ph.Model('graphene.ifc', divide_mass=False)
        elph = elphmod.elph.Model('graphene.epmatwp', 'graphene.wigner',
            el, ph, divide_mass=False)

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
