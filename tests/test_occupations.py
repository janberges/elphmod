#!/usr/bin/env python3

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

from elphmod import occupations
import numpy as np
import unittest

def d(f):
    def g(x):
        return f(x, 2.0)
    return g

double_fermi_dirac = d(occupations.double_fermi_dirac)
double_fermi_dirac.delta = d(occupations.double_fermi_dirac.delta)
double_fermi_dirac.delta_prime = d(occupations.double_fermi_dirac.delta_prime)
double_fermi_dirac.entropy = d(occupations.double_fermi_dirac.entropy)

tol = dict(rtol=1e-5, atol=1e-4)

class TestOccupations(unittest.TestCase):
    def test_derivatives(self):
        """Compare analytical and numerical derivatives of step functions."""

        x, dx = np.linspace(-10, 10, 2001, retstep=True)

        xd = x[1:] - dx / 2
        xdp = x[1:-1]
        xs = x + dx / 2
        xs0 = x[0] - dx / 2

        for f in [occupations.fermi_dirac, occupations.gauss,
                occupations.lorentz, occupations.marzari_vanderbilt,
                occupations.methfessel_paxton, double_fermi_dirac]:

            self.assertTrue(np.allclose(f.delta(xd),
                -np.diff(f(x)) / dx, **tol))

            self.assertTrue(np.allclose(f.delta_prime(xdp),
                -np.diff(f(x), 2) / dx ** 2, **tol))

            if hasattr(f, 'entropy'):
                self.assertTrue(np.allclose(f.entropy(xs) - f.entropy(xs0),
                    -dx * np.cumsum(x * f.delta(x)), **tol))

if __name__ == '__main__':
    unittest.main()
