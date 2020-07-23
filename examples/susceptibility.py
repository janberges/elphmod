#/usr/bin/env python

import elphmod
import matplotlib.pyplot as plt

k, x, GMKG = elphmod.bravais.GMKG(150, corner_indices=True)

el = elphmod.el.Model('data/NbSe2_hr.dat')

e = elphmod.dispersion.dispersion_full(el.H, 300)

chi = elphmod.diagrams.susceptibility(e[:, :, 0])
chi = elphmod.dispersion.dispersion(chi, k, broadcast=False)

if elphmod.MPI.comm.rank == 0:
    plt.xlabel('wave vector')
    plt.ylabel('susceptibility (1/eV)')
    plt.xticks(x[GMKG], 'GMKG')
    plt.plot(x, chi)
    plt.show()
