#!/usr/bin/env python3

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import elphmod.models.tas2
import numpy as np
import unittest

elphmod.misc.verbosity = 0

tol = dict(rtol=1e-2, atol=0.0)

class TestDiagrams(unittest.TestCase):
    def test_expansion(self, eps=1e-4, nk=(4, 4),
            kT=0.01, f=elphmod.occupations.fermi_dirac):
        """Compare lowest-order diagrams to finite differences."""

        k = elphmod.bravais.mesh(*nk)
        q = np.zeros((1, 2))

        el, ph, elph = elphmod.models.tas2.create(rydberg=True,
            divide_mass=False)

        g0 = elph.gR()
        g0[2, 0, 0] += elph.data.max() # ensure nonzero first-order term

        elph = elph.supercell(2, 2)

        H = elphmod.dispersion.sample(elph.el.H, k)

        e, U = np.linalg.eigh(H)

        H = e[..., np.newaxis] * np.eye(elph.el.size)

        u = 1 - 2 * np.random.rand(1, elph.ph.size, 1)

        elphmod.MPI.comm.Bcast(u)

        gu = elph.sample(q=q, U=U, u=u)[0]

        prefactor = 2 * kT / np.prod(nk)

        def E(H):
            x = np.linalg.eigvalsh(H) / kT

            return prefactor * ((f(x) * x).sum() - f.entropy(x).sum())

        diff = (E(H + eps * gu) - E(H - eps * gu)) / (2 * eps)

        pert = elphmod.diagrams.first_order(e, gu,
            kT=kT, occupations=f)[0].real

        self.assertTrue(np.allclose(diff, pert, **tol))

        diff = (E(H - eps * gu) - 2 * E(H) + E(H + eps * gu)) / eps ** 2

        pert = elphmod.diagrams.phonon_self_energy(q, e, g2=abs(gu) ** 2,
            kT=kT, occupations=f)[0, 0].real

        self.assertTrue(np.allclose(diff, pert, **tol))

        diff = (-E(H - 2 * eps * gu) / 2 + E(H - eps * gu) - E(H + eps * gu)
            + E(H + 2 * eps * gu) / 2) / eps ** 3

        pert = elphmod.diagrams.triangle(q[0], q[0], e, gu, gu, gu,
            kT=kT, occupations=f).real

        self.assertTrue(np.allclose(diff, pert, **tol))

if __name__ == '__main__':
    unittest.main()
