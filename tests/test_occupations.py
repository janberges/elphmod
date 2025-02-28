#!/usr/bin/env python3

# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

from elphmod import occupations
import numpy as np
import unittest

tol = dict(rtol=1e-5, atol=1e-4)

class TestOccupations(unittest.TestCase):
    def _test_derivatives(self, f, xmax=10.0, nx=2001):
        """Compare analytical and numerical derivatives of step functions."""

        x, dx = np.linspace(-xmax, xmax, nx, retstep=True)

        xd = x[1:] - dx / 2
        xdp = x[1:-1]
        xs = x + dx / 2
        xs0 = x[0] - dx / 2

        self.assertTrue(np.allclose(f.delta(xd),
            -np.diff(f(x)) / dx, **tol))

        self.assertTrue(np.allclose(f.delta_prime(xdp),
            -np.diff(f(x), 2) / dx ** 2, **tol))

        if hasattr(f, 'entropy'):
            self.assertTrue(np.allclose(f.entropy(xs) - f.entropy(xs0),
                -dx * np.cumsum(x * f.delta(x)), **tol))

    def test_derivatives_fermi_dirac(self):
        """Validate derivatives of Fermi-Dirac step function."""
        self._test_derivatives(occupations.fermi_dirac)

    def test_derivatives_gauss(self):
        """Validate derivatives of Gauss step function."""
        self._test_derivatives(occupations.gauss)

    def test_derivatives_lorentz(self):
        """Validate derivatives of Lorentz step function."""
        self._test_derivatives(occupations.lorentz)

    def test_derivatives_marzari_vanderbilt(self):
        """Validate derivatives of Marzari-Vanderbilt step function."""
        self._test_derivatives(occupations.marzari_vanderbilt)

    def test_derivatives_methfessel_paxton(self):
        """Validate derivatives of Methfessel-Paxton step function."""
        self._test_derivatives(occupations.methfessel_paxton)

    def test_derivatives_double_fermi_dirac(self):
        """Validate derivatives of double Fermi-Dirac step function."""
        self._test_derivatives(occupations.double_fermi_dirac)

if __name__ == '__main__':
    unittest.main()
