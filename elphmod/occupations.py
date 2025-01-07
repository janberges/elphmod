# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

"""Step and delta smearing functions."""

import numpy as np

try:
    from scipy.special import erf
except ImportError:
    import math

    erf = np.vectorize(math.erf)

xmax = 709.0 # approx. log([max. double] / 2 - 1)

def bose_einstein(x):
    """Calculate Bose-Einstein function."""

    x = np.array(x)
    x.real = np.minimum(x.real, xmax)

    return 1 / (np.exp(x) - 1)

def fermi_dirac(x):
    """Calculate Fermi function."""

    # return 1 - 0.5 * np.tanh(0.5 * x)

    x = np.array(x)
    x.real = np.minimum(x.real, xmax)

    return 1 / (np.exp(x) + 1)

def fermi_dirac_delta(x):
    """Calculate negative derivative of Fermi function."""

    x = np.array(x)
    x.real = np.minimum(abs(x.real), xmax)

    return 1 / (2 * np.cosh(x) + 2)

def fermi_dirac_delta_prime(x):
    """Calculate negative 2nd derivative of Fermi function."""

    x = np.array(x)
    x.real = np.sign(x.real) * np.minimum(abs(x.real), xmax)

    return -np.sinh(x) / (2 * (np.cosh(x) + 1) ** 2)

def fermi_dirac_entropy(x):
    """Calculate electronic entropy."""

    x = np.array(x)
    x.real = np.sign(x.real) * np.minimum(abs(x.real), xmax)

    f = fermi_dirac(x)

    return x * (f - 1) - np.log(f)

fermi_dirac.delta = fermi_dirac_delta
fermi_dirac.delta_prime = fermi_dirac_delta_prime
fermi_dirac.entropy = fermi_dirac_entropy

def gauss(x):
    """Calculate Gaussian step function."""

    return (1 - erf(x)) / 2

def gauss_delta(x):
    """Calculate negative derivative of Gaussian step function."""

    return np.exp(-x * x) / np.sqrt(np.pi)

def gauss_delta_prime(x):
    """Calculate negative 2nd derivative of Gaussian step function."""

    return -2 * x * np.exp(-x * x) / np.sqrt(np.pi)

def gauss_entropy(x):
    """Calculate Gaussian generalized electronic entropy."""

    return gauss_delta(x) / 2

gauss.delta = gauss_delta
gauss.delta_prime = gauss_delta_prime
gauss.entropy = gauss_entropy

def marzari_vanderbilt(x):
    """Calculate Marzari-Vanderbilt (cold smearing) step function."""

    y = x + 1 / np.sqrt(2)

    return (1 - erf(y)) / 2 + np.exp(-y * y) / np.sqrt(2 * np.pi)

def marzari_vanderbilt_delta(x):
    """Calculate negative derivative of Marzari-Vanderbilt step function."""

    y = x + 1 / np.sqrt(2)

    return (np.sqrt(2) * y + 1) * np.exp(-y * y) / np.sqrt(np.pi)

def marzari_vanderbilt_delta_prime(x):
    """Calculate negative 2nd derivative of Marzari-Vanderbilt step function."""

    y = x + 1 / np.sqrt(2)

    return (np.sqrt(2) - 2 * y
        * (np.sqrt(2) * y + 1)) * np.exp(-y * y) / np.sqrt(np.pi)

def marzari_vanderbilt_entropy(x):
    """Calculate Marzari-Vanderbilt generalized electronic entropy."""

    y = x + 1 / np.sqrt(2)

    return y * np.exp(-y * y) / np.sqrt(2 * np.pi)

marzari_vanderbilt.delta = marzari_vanderbilt_delta
marzari_vanderbilt.delta_prime = marzari_vanderbilt_delta_prime
marzari_vanderbilt.entropy = marzari_vanderbilt_entropy

def hermite_polynomials(x, N=100):
    r"""Generate Hermite polynomials.

    .. math::

        H_0(x) &= 1 \\
        H_1(x) &= 2 x \\
        H_{n + 1}(x) &= 2 x H_n(x) - 2 n H_{n - 1}(x)
    """
    H = 0
    h = np.ones_like(x)

    for n in range(N):
        yield h

        h, H = 2 * x * h - 2 * n * H, h

def methfessel_paxton_term(x, order=1, diff=0):
    r"""Calculate Methfessel-Paxton term (without zeroth order).

    From Phys. Rev. B 40, 3616 (1989):

    .. math::

        S_0(x) &= \frac {1 - erf(x)} 2 \\
        S_N(x) &= S_0(x) + \sum_{n = 1}^N A_n H_{2 n - 1}(x) \exp(-x^2) \\
        D_N(x) &= -S'(N, x) = \sum{n = 0}^N A_n H_{2 n}(x) \exp(-x^2) \\
        A_n &= \frac{(-1)^n}{\sqrt \pi n! 4^n}

    This routine has been adapted from Quantum ESPRESSO:

    * Step function: Modules/wgauss.f90
    * Delta function: Modules/w0gauss.f90
    """
    a = 1.0
    s = 0.0
    h = hermite_polynomials(x)

    for n in range(diff):
        a *= -1
        next(h)

    for n in range(1, order + 1):
        a /= -4 * n
        next(h)

        s += a * next(h)

    return s * gauss_delta(x)

def methfessel_paxton(x):
    """Calculate Methfessel-Paxton step function."""

    return gauss(x) + methfessel_paxton_term(x,
        order=methfessel_paxton.order, diff=0)

def methfessel_paxton_delta(x):
    """Calculate negative derivative of Methfessel-Paxton step function."""

    return gauss.delta(x) - methfessel_paxton_term(x,
        order=methfessel_paxton.order, diff=1)

def methfessel_paxton_delta_prime(x):
    """Calculate negative 2nd derivative of Methfessel-Paxton step function."""

    return gauss.delta_prime(x) - methfessel_paxton_term(x,
        order=methfessel_paxton.order, diff=2)

def methfessel_paxton_entropy(x):
    """Calculate Methfessel-Paxton generalized electronic entropy."""

    if methfessel_paxton.order != 1:
        raise NotImplementedError('MP entropy only implemented for 1st order.')

    return -0.5 * methfessel_paxton_term(x,
        order=methfessel_paxton.order, diff=1)

methfessel_paxton.order = 1
methfessel_paxton.delta = methfessel_paxton_delta
methfessel_paxton.delta_prime = methfessel_paxton_delta_prime
methfessel_paxton.entropy = methfessel_paxton_entropy

def double_fermi_dirac(x):
    """Calculate double Fermi function."""

    return (fermi_dirac(x - double_fermi_dirac.d)
          + fermi_dirac(x + double_fermi_dirac.d)) / 2

def double_fermi_dirac_delta(x):
    """Calculate negative derivative of double Fermi function."""

    return (fermi_dirac.delta(x - double_fermi_dirac.d)
          + fermi_dirac.delta(x + double_fermi_dirac.d)) / 2

def double_fermi_dirac_delta_prime(x):
    """Calculate negative 2nd derivative of double Fermi function."""

    return (fermi_dirac.delta_prime(x - double_fermi_dirac.d)
          + fermi_dirac.delta_prime(x + double_fermi_dirac.d)) / 2

def double_fermi_dirac_entropy(x):
    """Calculate double-Fermi-Dirac generalized electronic entropy."""

    return (fermi_dirac.entropy(x - double_fermi_dirac.d)
          + fermi_dirac.entropy(x + double_fermi_dirac.d)
        + double_fermi_dirac.d * (fermi_dirac(x - double_fermi_dirac.d)
                                - fermi_dirac(x + double_fermi_dirac.d))) / 2

double_fermi_dirac.d = 5.0
double_fermi_dirac.delta = double_fermi_dirac_delta
double_fermi_dirac.delta_prime = double_fermi_dirac_delta_prime
double_fermi_dirac.entropy = double_fermi_dirac_entropy

def two_fermi_dirac(x):
    """Calculate two Fermi functions."""

    dy = (x < 0) * two_fermi_dirac.d
    dx = 2 * dy - two_fermi_dirac.d

    return fermi_dirac(x + dx)

def two_fermi_dirac_delta(x):
    """Calculate negative derivative of two Fermi functions."""

    dy = (x < 0) * two_fermi_dirac.d
    dx = 2 * dy - two_fermi_dirac.d

    return fermi_dirac.delta(x + dx)

def two_fermi_dirac_delta_prime(x):
    """Calculate negative 2nd derivative of two Fermi functions."""

    dy = (x < 0) * two_fermi_dirac.d
    dx = 2 * dy - two_fermi_dirac.d

    return fermi_dirac.delta_prime(x + dx)

def two_fermi_dirac_entropy(x):
    """Calculate generalized electronic entropy for two Fermi functions."""

    dy = (x < 0) * two_fermi_dirac.d
    dx = 2 * dy - two_fermi_dirac.d

    return fermi_dirac.entropy(x + dx) - dx * fermi_dirac(x + dx) + dy

two_fermi_dirac.d = 5.0
two_fermi_dirac.delta = two_fermi_dirac_delta
two_fermi_dirac.delta_prime = two_fermi_dirac_delta_prime
two_fermi_dirac.entropy = two_fermi_dirac_entropy

def lorentz(x):
    r"""Calculate Lorentz step function.

    Used to simulate the influence of a wide box-shaped hybridization function
    at low temperatures. Formula derived by Tim O. Wehling and Erik G.C.P. van
    Loon. Here, we have :math:`x = \epsilon / h` with the height :math:`h` of
    the hybridization, instead of :math:`x = \epsilon / k T` with the
    temperature :math:`T`.
    """
    return 0.5 - np.arctan(x / np.pi) / np.pi

def lorentz_delta(x):
    """Calculate negative derivative of Lorentz step function."""

    return 1 / (x * x + np.pi * np.pi)

def lorentz_delta_prime(x):
    """Calculate negative 2nd derivative of Lorentz step function."""

    return -2 * x / (x * x + np.pi * np.pi) ** 2

lorentz.delta = lorentz_delta
lorentz.delta_prime = lorentz_delta_prime

def heaviside(x):
    """Calculate (reflected) Heaviside function."""

    return 0.5 - 0.5 * np.sign(x)

def heaviside_delta(x):
    """Calculate negative derivative of (reflected) Heaviside function."""

    delta = np.copy(x)

    zero = delta == 0

    delta[zero] = np.inf
    delta[~zero] = 0.0

    return delta

def heaviside_delta_prime(x):
    """Calculate negative 2nd derivative of (reflected) Heaviside function."""

    delta_prime = np.copy(x)

    zero = delta_prime == 0

    delta_prime[zero] = -np.copysign(np.inf, delta_prime[zero])
    delta_prime[~zero] = 0.0

    return delta_prime

def heaviside_entropy(x):
    """Calculate negative Heaviside generalized electronic entropy."""

    return np.zeros_like(x)

heaviside.delta = heaviside_delta
heaviside.delta_prime = heaviside_delta_prime
heaviside.entropy = heaviside_entropy

def fermi_dirac_matsubara(x, nmats=1000):
    """Calculate Fermi function as Matsubara sum."""

    inu = 1j * (2 * np.arange(nmats) + 1) * np.pi

    return 0.5 + 2 * np.sum(np.subtract.outer(inu, x) ** -1, axis=0).real

def fermi_dirac_matsubara_delta(x, nmats=1000):
    """Calculate negative derivative of Fermi function as Matsubara sum."""

    inu = 1j * (2 * np.arange(nmats) + 1) * np.pi

    return -2 * np.sum(np.subtract.outer(inu, x) ** -2, axis=0).real

def fermi_dirac_matsubara_delta_prime(x, nmats=1000):
    """Calculate negative 2nd derivative of Fermi function as Matsubara sum."""

    inu = 1j * (2 * np.arange(nmats) + 1) * np.pi

    return -4 * np.sum(np.subtract.outer(inu, x) ** -3, axis=0).real

fermi_dirac_matsubara.delta = fermi_dirac_matsubara_delta
fermi_dirac_matsubara.delta_prime = fermi_dirac_matsubara_delta_prime

def smearing(smearing='gaussian', **ignore):
    """Select smearing function via name used in Quantum ESPRESSO.

    Parameters
    ----------
    smearing : str, default 'gaussian'
        Any available option for PWscf input parameter ``smearing``, as well as
        ``lorentzian``, ``lorentz``, and ``heaviside``. If no string is passed,
        it is returned as is.
    **ignore
        Ignored keyword arguments, e.g., parameters from 'func'`read_pwi`.

    Returns
    -------
    function
        Smearing function.
    """
    if not isinstance(smearing, str):
        return smearing

    name = smearing.lower()

    if name in {'gaussian', 'gauss'}:
        return gauss
    if name in {'methfessel-paxton', 'm-p', 'mp'}:
        return methfessel_paxton
    if name in {'marzari-vanderbilt', 'cold', 'm-v', 'mv'}:
        return marzari_vanderbilt
    if name in {'fermi-dirac', 'f-d', 'fd'}:
        return fermi_dirac
    if name in {'lorentzian', 'lorentz'}:
        return lorentz
    if name in {'heaviside'}:
        return heaviside

def find_Fermi_level(n, e, kT=0.025, f='fd', mu=None, tol=1e-5, eps=1e-10):
    """Determine chemical potential via fixed-point iteration.

    See Eqs. 4.21 and 4.22 of https://janberges.de/theses/Master_Jan_Berges.pdf.

    Parameters
    ----------
    n : float
        Number of electrons (with spin) per unit cell.
    e : ndarray
        Electronic energies for representative k points.
    kT : float
        Smearing temperature.
    f : function
        Electron distribution as a function of energy divided by `kT`.
    mu : float
        Initial guess for chemical potential. By default, an estimate based on
        the assumption of a constant density of states is used.
    tol : float
        Tolerance for the number of electrons.
    eps : float
        Smallest allowed absolute value of divisor.

    Returns
    -------
    float
        Chemical potential.
    """
    f = smearing(f)

    # map number of electrons to whole system and spinless electrons:

    scale = 0.5 * e.size / e.shape[-1]
    n *= scale
    tol *= scale

    # make initial guess for chemical potential:

    if mu is None:
        mu = (e.min() * (e.size - n) + e.max() * n) / e.size

    # solve fixed-point equation:

    while True:
        xi = e - mu
        f0 = f(0.0)
        fx = f(xi / kT)
        w = f0 - fx

        ok = abs(xi) > eps
        w[ok] /= xi[ok]
        w[~ok] = f.delta(0.0) / kT

        if abs(fx.sum() - n) < tol:
            return mu

        mu = (n - e.size * f0 + (e * w).sum()) / w.sum()

def find_Fermi_level_simple(n, e, kT=0.025, f='fd', mu=None, tol=1e-5,
        damp=1e-2):
    """Determine chemical potential via simple fixed-point iteration.

    Parameters
    ----------
    n : float
        Number of electrons (with spin) per unit cell.
    e : ndarray
        Electronic energies for representative k points.
    kT : float
        Smearing temperature.
    f : function
        Electron distribution as a function of energy divided by `kT`.
    mu : float
        Initial guess for chemical potential. By default, an estimate based on
        the assumption of a constant density of states is used.
    tol : float
        Tolerance for the number of electrons.
    damp : float
        Damping factor in the fixed-point equation. Large values may prevent
        convergence; small values will slow down convergence.

    Returns
    -------
    float
        Chemical potential.
    """
    f = smearing(f)

    # map number of electrons to whole system and spinless electrons:

    scale = 0.5 * e.size / e.shape[-1]
    n *= scale
    tol *= scale

    # make initial guess for chemical potential:

    if mu is None:
        mu = (e.min() * (e.size - n) + e.max() * n) / e.size

    # solve fixed-point equation:

    while True:
        N = f((e - mu) / kT).sum()

        if abs(N - n) < tol:
            return mu

        mu += (n / N - 1) * damp
