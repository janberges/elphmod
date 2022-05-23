#!/usr/bin/env python3

# Copyright (C) 2017-2022 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import matplotlib.pyplot as plt
import model
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

g = elph.sample(q=q, U=U)

info('Calculate retarded displacement-displacement correlation function..')

w = np.linspace(w0.min(), w0.max(), len(q))

Pi0 = elphmod.diagrams.phonon_self_energy(q, e, g=g, kT=kT, occupations=f)

Piw = elphmod.diagrams.phonon_self_energy(q, e, g=g, kT=kT, occupations=f,
    omega=w + 1j * eta1)

Dw = D0[..., None] - Pi0[..., None] + Piw

A = elphmod.ph.spectral_function(Dw, w, eta2)

info('Plot results..')

A = elphmod.plot.adjust_pixels(A, GMKG, x[GMKG], width, height)

if comm.rank == 0:
    plt.imshow(A[::-1], extent=(x[0], x[-1], w[0], w[-1]), cmap='ocean_r')
    plt.plot(x, w0, 'k')
    plt.ylabel('Phonon energy (Ry)')
    plt.xlabel('Wave vector')
    plt.xticks(x[GMKG], 'GMKG')
    plt.axis('auto')
    plt.show()
