#!/usr/bin/env python3

# Copyright (C) 2017-2023 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import data.graphene
import elphmod
import matplotlib.pyplot as plt
import numpy as np

comm = elphmod.MPI.comm
info = elphmod.MPI.info

info('Define parameters')

nk = 60
width = 300
height = 300

kT = 0.01
f = elphmod.occupations.fermi_dirac

eta1 = 0.001 # broadening of self-energy
eta2 = 0.0001 # broadening of all modes

info('Load model Hamiltonian for graphene..')

el = elphmod.el.Model('data/graphene')
ph = elphmod.ph.Model('data/graphene.ifc')
elph = elphmod.elph.Model('data/graphene.epmatwp', 'data/graphene.wigner',
    el, ph)

el.data /= elphmod.misc.Ry

info('Sample and diagonalize model Hamiltonian..')

q, x, GMKG = elphmod.bravais.GMKG(nk, corner_indices=True, mesh=True,
    straight=False, lift_degen=False)

e, U = elphmod.dispersion.dispersion_full_nosym(el.H, nk, vectors=True)

D0 = elphmod.dispersion.sample(ph.D, q)
w0 = elphmod.ph.sgnsqrt(np.linalg.eigvalsh(D0))

g = elph.sample(q, U=U)

info('Calculate retarded displacement-displacement correlation function..')

w, dw = np.linspace(w0.min(), w0.max(), len(q), retstep=True)

Pi0 = elphmod.diagrams.phonon_self_energy(q, e, g=g, kT=kT, occupations=f)

Piw = elphmod.diagrams.phonon_self_energy(q, e, g=g, kT=kT, occupations=f,
    omega=w + 1j * eta1)

Dw = D0[..., None] - Pi0[..., None] + Piw

A = elphmod.ph.spectral_function(Dw, w, eta2)

info('Plot results..')

integral = A.sum(axis=0) * dw

A = elphmod.plot.adjust_pixels(A, GMKG, x[GMKG], width, height)

if comm.rank == 0:
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
        gridspec_kw=dict(height_ratios=(3, 1)))

    ax1.set_ylabel('Phonon energy (Ry)')
    ax2.set_ylabel('Integrated phonon spectral function')
    ax2.set_xlabel('Wave vector')
    ax2.set_xticks(x[GMKG])
    ax2.set_xticklabels('GMKG')

    ax1.imshow(A[::-1], extent=(x[0], x[-1], w[0], w[-1]), cmap='ocean_r')
    ax1.plot(x, w0, 'k')
    ax1.axis('auto')

    ax2.plot(x, integral)
    ax2.set_ylim(0.0, 2 * ph.size)

    plt.show()
