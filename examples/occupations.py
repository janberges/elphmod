#!/usr/bin/env python3

from elphmod import MPI, occupations
import matplotlib.pyplot as plt
import numpy as np

if MPI.comm.rank:
    raise SystemExit

x, dx = np.linspace(-10, 10, 2001, retstep=True)

d = 2.0

style = dict(color='lightgray', linestyle='dashed')

print('Plot step functions..')

plt.axvline(x=0.0, **style)
plt.axhline(y=0.0, **style)
plt.axhline(y=0.5, **style)
plt.axhline(y=1.0, **style)

plt.plot(x, occupations.fermi_dirac(x), label='Fermi-Dirac')
plt.plot(x, occupations.gauss(x), label='Gauss')
plt.plot(x, occupations.marzari_vanderbilt(x), label='Marzari-Vanderbilt')
plt.plot(x, occupations.methfessel_paxton(x), label='Methfessel-Paxton')
plt.plot(x, occupations.double_fermi_dirac(x, d), label='Double Fermi-Dirac')
plt.plot(x, occupations.lorentz(x), label='Lorentz')

plt.xlabel(r'$x$')
plt.ylabel(r'$f(x)$')
plt.legend()
plt.show()

print('Plot delta functions..')

plt.axvline(x=0.0, **style)
plt.axhline(y=0.0, **style)

plt.plot(x, occupations.fermi_dirac.delta(x), label='Fermi-Dirac')
plt.plot(x, occupations.gauss.delta(x), label='Gauss')
plt.plot(x, occupations.marzari_vanderbilt.delta(x), label='Marzari-Vanderbilt')
plt.plot(x, occupations.methfessel_paxton.delta(x), label='Methfessel-Paxton')
plt.plot(x, occupations.double_fermi_dirac.delta(x, d),
    label='Double Fermi-Dirac')
plt.plot(x, occupations.lorentz.delta(x), label='Lorentz')

X = (x[1:] + x[:-1]) / 2
plt.plot(X, -np.diff(occupations.fermi_dirac(x)) / dx, 'k--')
plt.plot(X, -np.diff(occupations.gauss(x)) / dx, 'k--')
plt.plot(X, -np.diff(occupations.marzari_vanderbilt(x)) / dx, 'k--')
plt.plot(X, -np.diff(occupations.methfessel_paxton(x)) / dx, 'k--')
plt.plot(X, -np.diff(occupations.double_fermi_dirac(x, d)) / dx, 'k--')
plt.plot(X, -np.diff(occupations.lorentz(x)) / dx, 'k--')

plt.xlabel(r'$x$')
plt.ylabel(r'$\delta(x)$')
plt.legend()
plt.show()

print('Plot derivatives of delta functions..')

plt.axvline(x=0.0, **style)
plt.axhline(y=0.0, **style)

plt.plot(x, occupations.fermi_dirac.delta_prime(x), label='Fermi-Dirac')
plt.plot(x, occupations.gauss.delta_prime(x), label='Gauss')
plt.plot(x, occupations.marzari_vanderbilt.delta_prime(x),
    label='Marzari-Vanderbilt')
plt.plot(x, occupations.methfessel_paxton.delta_prime(x),
    label='Methfessel-Paxton')
plt.plot(x, occupations.double_fermi_dirac.delta_prime(x, d),
    label='Double Fermi-Dirac')
plt.plot(x, occupations.lorentz.delta_prime(x), label='Lorentz')

X = (X[1:] + X[:-1]) / 2
plt.plot(X, -np.diff(occupations.fermi_dirac(x), 2) / dx ** 2, 'k--')
plt.plot(X, -np.diff(occupations.gauss(x), 2) / dx ** 2, 'k--')
plt.plot(X, -np.diff(occupations.marzari_vanderbilt(x), 2) / dx ** 2, 'k--')
plt.plot(X, -np.diff(occupations.methfessel_paxton(x), 2) / dx ** 2, 'k--')
plt.plot(X, -np.diff(occupations.double_fermi_dirac(x, d), 2) / dx ** 2, 'k--')
plt.plot(X, -np.diff(occupations.lorentz(x), 2) / dx ** 2, 'k--')

plt.xlabel(r'$x$')
plt.ylabel(r"$\delta'(x)$")
plt.legend()
plt.show()

print('Plot generalized entropy..')

plt.axvline(x=0.0, **style)
plt.axhline(y=0.0, **style)

plt.plot(x, occupations.fermi_dirac.entropy(x), label='Fermi-Dirac')
plt.plot(x, occupations.gauss.entropy(x), label='Gauss')
plt.plot(x, occupations.marzari_vanderbilt.entropy(x),
    label='Marzari-Vanderbilt')
plt.plot(x, occupations.methfessel_paxton.entropy(x), label='Methfessel-Paxton')
plt.plot(x, occupations.double_fermi_dirac.entropy(x, d),
    label='Double Fermi-Dirac')

plt.plot(x, -dx * np.cumsum(x * occupations.fermi_dirac.delta(x)), 'k--')
plt.plot(x, -dx * np.cumsum(x * occupations.gauss.delta(x)), 'k--')
plt.plot(x, -dx * np.cumsum(x * occupations.marzari_vanderbilt.delta(x)), 'k--')
plt.plot(x, -dx * np.cumsum(x * occupations.methfessel_paxton.delta(x)), 'k--')
plt.plot(x, -dx * np.cumsum(x * occupations.double_fermi_dirac.delta(x, d)),
    'k--')

plt.xlabel(r'$x$')
plt.ylabel(r'$-\int_{-\infty}^x y \delta(y) dy$')
plt.legend()
plt.show()
