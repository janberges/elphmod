#!/usr/bin/env python3

from elphmod import MPI, occupations
import matplotlib.pyplot as plt
import numpy as np

if MPI.comm.rank:
    raise SystemExit

x, dx = np.linspace(-5, 5, 1000, retstep=True)

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
plt.plot(x, occupations.lorentz(x), label='Lorentz')

plt.xlabel = r'$x$'
plt.ylabel = r'$f(x)$'
plt.legend()
plt.show()

print('Plot delta functions..')

plt.axvline(x=0.0, **style)
plt.axhline(y=0.0, **style)

plt.plot(x, occupations.fermi_dirac.delta(x), label='Fermi-Dirac')
plt.plot(x, occupations.gauss.delta(x), label='Gauss')
plt.plot(x, occupations.marzari_vanderbilt.delta(x), label='Marzari-Vanderbilt')
plt.plot(x, occupations.methfessel_paxton.delta(x), label='Methfessel-Paxton')
plt.plot(x, occupations.lorentz.delta(x), label='Lorentz')

X = (x[1:] + x[:-1]) / 2
plt.plot(X, -np.diff(occupations.fermi_dirac(x)) / dx, 'k--')
plt.plot(X, -np.diff(occupations.gauss(x)) / dx, 'k--')
plt.plot(X, -np.diff(occupations.marzari_vanderbilt(x)) / dx, 'k--')
plt.plot(X, -np.diff(occupations.methfessel_paxton(x)) / dx, 'k--')
plt.plot(X, -np.diff(occupations.lorentz(x)) / dx, 'k--')

plt.xlabel = r'$x$'
plt.ylabel = r'$\delta(x)$'
plt.legend()
plt.show()
