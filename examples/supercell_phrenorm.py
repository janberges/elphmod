#!/usr/bin/env python3

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

# This example shows that phonon renormalization and supercell mapping commute.

import copy
import elphmod
import elphmod.models.graphene
import matplotlib.pyplot as plt
import numpy as np

N = 2

f = elphmod.occupations.fermi_dirac
kT = 0.01
mu = elphmod.models.graphene.t / elphmod.misc.Ry

nq = 6
nk = 12

elphmod.models.graphene.create('data/graphene')

el = elphmod.el.Model('data/graphene')
el.data /= elphmod.misc.Ry
ph = elphmod.ph.Model('data/graphene.ifc')
elph = elphmod.elph.Model('data/graphene.epmatwp', 'data/graphene.wigner',
    el, ph)

q = 2 * np.pi * np.array([(q1, q2)
    for q1 in range(nq)
    for q2 in range(nq)], dtype=float) / nq

e, u = elphmod.dispersion.dispersion_full_nosym(elph.el.H, nk, vectors=True)
e -= mu
d = elphmod.dispersion.sample(elph.ph.D, q)
g = elph.sample(q, U=u)

d += elphmod.diagrams.phonon_self_energy(q, e, g=g, kT=kT, occupations=f)
d[0] += elphmod.diagrams.phonon_self_energy_fermi_shift(e, g[0], kT, f)

phrenorm = copy.deepcopy(ph)
elphmod.ph.q2r(phrenorm, D_full=d, nq=(nq, nq))

phrenorm = phrenorm.supercell(N, N)

Nq = nq // N
Nk = nk // N

ElPh = elph.supercell(N, N)

Q = 2 * np.pi * np.array([(q1, q2)
    for q1 in range(Nq)
    for q2 in range(Nq)], dtype=float) / Nq

E, U = elphmod.dispersion.dispersion_full_nosym(ElPh.el.H, Nk, vectors=True)
E -= mu
D = elphmod.dispersion.sample(ElPh.ph.D, Q)
G = ElPh.sample(Q, U=U)

D += elphmod.diagrams.phonon_self_energy(Q, E, g=G, kT=kT, occupations=f)
D[0] += elphmod.diagrams.phonon_self_energy_fermi_shift(E, G[0], kT, f)

Phrenorm = copy.deepcopy(ElPh.ph)
elphmod.ph.q2r(Phrenorm, D_full=D, nq=(Nq, Nq))

path = 'GMKG'
q_path, x, corners = elphmod.bravais.path(path, ibrav=4, N=300)

w = elphmod.ph.sgnsqrt(elphmod.dispersion.dispersion(phrenorm.D, q_path))
W = elphmod.ph.sgnsqrt(elphmod.dispersion.dispersion(Phrenorm.D, q_path))

if elphmod.MPI.comm.rank == 0:
    plt.plot(x, w, 'k')
    plt.plot(x, W, 'c:')

    plt.ylabel('Phonon energy (meV)')
    plt.xlabel('Wave vector')
    plt.xticks(x[corners], path)
    plt.show()
