#/usr/bin/env python3

import elphmod
import matplotlib.pyplot as plt

Ry2eV = 13.605693009

q, x, GMKG = elphmod.bravais.GMKG(corner_indices=True)

for method in 'dfpt', 'cdfpt':
    ph = elphmod.ph.Model('%s.ifc' % method, apply_asr=True)

    w2, u, order = elphmod.dispersion.dispersion(ph.D, q,
        vectors=True, order=True)

    w = elphmod.ph.sgnsqrt(w2) * Ry2eV * 1e3

    if elphmod.MPI.comm.rank == 0:
        proj = elphmod.ph.polarization(u, q)

        for nu in range(ph.size):
            fatbands = elphmod.plot.compline(x, w[:, nu], proj[:, nu])

            for fatband, color in zip(fatbands, 'ymk'):
                plt.fill(*fatband, color=color, linewidth=0.0)

        plt.ylabel(r'$\omega$ (meV)')
        plt.xlabel(r'$q$')
        plt.xticks(x[GMKG], 'GMKG')
        plt.show()
