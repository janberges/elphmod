#!/usr/bin/env python3

from elphmod import MPI, occupations
import matplotlib.pyplot as plt
import numpy as np

if MPI.comm.rank:
    raise SystemExit

x = np.linspace(-5, 5, 1000)

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

plt.xlabel = r'$x$'
plt.ylabel = r'$\delta(x)$'
plt.legend()
plt.show()
