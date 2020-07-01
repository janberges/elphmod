#/usr/bin/env python3

import elphmod
import matplotlib.pyplot as plt

k, x, GMKG = elphmod.bravais.GMKG(100, corner_indices=True)

el = elphmod.el.Model('TaS2_hr.dat')
mu = elphmod.el.read_Fermi_level('scf.out')

e, U, order = elphmod.dispersion.dispersion(el.H, k,
    vectors=True, order=True)
e -= mu

if elphmod.MPI.comm.rank == 0:
    proj = 0.1 * abs(U) ** 2

    for n in range(el.size):
        fatbands = elphmod.plot.compline(x, e[:, n], proj[:, :, n])

        for fatband, color in zip(fatbands, 'rgb'):
            plt.fill(*fatband, color=color, linewidth=0.0)

    plt.ylabel(r'$\epsilon$ (eV)')
    plt.xlabel(r'$k$')
    plt.xticks(x[GMKG], 'GMKG')
    plt.show()
