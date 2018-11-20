#/usr/bin/env python

import math
import numpy as np

xmax = 709.0 # approx. log([max. double] / 2 - 1)

def fermi_dirac(x):
    """Calculate Fermi function."""

    x = np.minimum(x, xmax)

    return 1 / (np.exp(x) + 1)

def fermi_dirac_delta(x):
    """Calculate negative derivative of Fermi function."""

    x = np.minimum(np.absolute(x), xmax)

    return 1 / (2 * (np.cosh(x) + 1))

def methfessel_paxton_general(x, N=0):
    """Calculate Methfessel-Paxton step function [Phys. Rev. B 40, 3616 (1989)].

        S(0, x) = 1/2 [1 - erf(x)]
        S(n, x) = S(0, x) + sum[n = 1, N] A(n) H(2 n - 1, x) exp(-x^2)
        A(n) = (-1)^n / [sqrt(pi) n! 4^n]

    Hermite polynomials:

        H(0,     x) = 1
        H(1,     x) = 2 x
        H(n + 1, x) = 2 x H(n, x) - 2 n H(n - 1, x)

    For N = 0, the Gaussian step function is returned.
    This routine has been adapted from Quantum ESPRESSO (Modules/wgauss.f90)."""

    S = 0.5 * (1 - math.erf(x))

    if N > 0:
        H = 0              # H(-1, x)
        h = np.exp(-x * x) # H( 0, x) [actually 1, but our H contains exp(-x^2)]

        a = 1 / np.sqrt(np.pi)

        m = 0
        for n in range(1, N + 1):
            H = 2 * x * h - 2 * m * H # H(1, x), H(3, x), ...
            m += 1

            a /= -4 * n
            S += a * H

            h = 2 * x * H - 2 * m * h # H(2, x), H(4, x), ...
            m += 1

    return S

methfessel_paxton_general = np.vectorize(methfessel_paxton_general)

def gauss(x):
    """Calculate Gaussian step function."""

    return methfessel_paxton_general(x, N=0)

def methfessel_paxton(x):
    """Calculate first-order Methfessel-Paxton step function."""

    return methfessel_paxton_general(x, N=1)
