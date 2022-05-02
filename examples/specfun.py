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

cmap = elphmod.plot.colormap(
    (0.0, elphmod.plot.Color(255, 255, 255)),
    (0.1, elphmod.plot.Color(0, 0, 255)),
    (0.2, elphmod.plot.Color(255, 0, 0)),
    (1.0, elphmod.plot.Color(0, 0, 0)),
    )

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

g = elphmod.elph.sample(elph.g, q, nk, U, broadcast=False)

g2 = elphmod.MPI.SharedArray((len(q), ph.size, ph.size,
    nk, nk, el.size, el.size), dtype=complex)

if comm.rank == 0:
    g2[...] = np.einsum('qiklmn,qjklmn->qijklmn', g.conj(), g)

g2.Bcast()

info('Calculate retarded displacement-displacement correlation function..')

w = np.linspace(w0.min(), w0.max(), len(q))

Pi0 = elphmod.diagrams.phonon_self_energy(q, e, g2, kT,
    occupations=f).reshape((len(q), ph.size, ph.size))

Piw = elphmod.diagrams.phonon_self_energy(q, e, g2, kT, occupations=f,
    omega=w + 1j * eta1).reshape((len(q), ph.size, ph.size, len(w)))

Dw = D0[..., None] - Pi0[..., None] + Piw

A = elphmod.ph.spectral_function(Dw, w, eta2)

info('Plot results..')

A = elphmod.plot.adjust_pixels(A, GMKG, x[GMKG], width, height)

A = elphmod.plot.color(A[::-1], minimum=0.0, cmap=cmap).astype(int)

if comm.rank == 0:
    w0 *= elphmod.misc.Ry * 1e3

    plt.imshow(A, extent=(x.min(), x.max(), w0.min(), w0.max()))
    plt.axis('auto')
    plt.plot(x, w0, 'k')
    plt.ylabel('Phonon energy (meV)')
    plt.xlabel('Wave vector')
    plt.xticks(x[GMKG], 'GMKG')
    plt.show()
