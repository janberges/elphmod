#!/usr/bin/env python3

import copy
import elphmod
import matplotlib.pyplot as plt
import numpy as np

comm = elphmod.MPI.comm
info = elphmod.MPI.info

Ry2eV = 13.605693009

kT = 0.005 * Ry2eV

nk = 4
nq = 2

nel = 1
nph = 9

info('Prepare wave vectors')

q = sorted(elphmod.bravais.irreducibles(nq))
q = 2 * np.pi * np.array(q, dtype=float) / nq

k, x, GMKG = elphmod.bravais.GMKG(100, corner_indices=True)

info('Prepare electrons')

el = elphmod.el.Model('TaS2_hr.dat')
mu = elphmod.el.read_Fermi_level('scf.out')

e, U = elphmod.dispersion.dispersion_full_nosym(el.H, nk, vectors=True)

e = e[..., :nel] - mu
U = U[..., :nel]

info('Prepare phonons')

ph = dict()

for method in 'cdfpt', 'dfpt':
    ph[method] = elphmod.ph.Model('%s.ifc' % method, apply_asr=True)

info('Prepare electron-phonon coupling')

g = dict()

for method in sorted(ph):
    elph = elphmod.elph.Model('%s.epmatwp1' % method, 'wigner.dat',
        el, ph['cdfpt'])

    g[method] = elph.sample(q=q, nk=nk, U=U, u=None, broadcast=False)

if comm.rank == 0:
    g2 = np.einsum('qiklnm,qjklnm->qijklnm', g['cdfpt'].conj(), g['dfpt'])
    g2 *= Ry2eV ** 3

    g2 += np.einsum('qijklnm->qjiklnm', g2.conj())
    g2 /= 2

else:
    g2 = np.empty((len(q), nph, nph, nk, nk, nel, nel), dtype=complex)

comm.Bcast(g2)

info('Calculate phonon self-energy')

Pi = elphmod.diagrams.phonon_self_energy(q, e, g2, kT=kT,
    occupations=elphmod.occupations.fermi_dirac)

Pi = np.reshape(Pi, (len(q), nph, nph))
Pi /= Ry2eV ** 2

info('Calculate bare dynamical matrices')

sizes, bounds = elphmod.MPI.distribute(len(q), bounds=True)

D    = np.empty((len(q),           nph, nph), dtype=complex)
my_D = np.empty((sizes[comm.rank], nph, nph), dtype=complex)

for my_iq, iq in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
    my_D[my_iq] = ph['cdfpt'].D(*q[iq])

comm.Allgatherv(my_D, (D, sizes * nph * nph))

info('Renormalize phonons')

ph['pp'] = copy.copy(ph['cdfpt'])

elphmod.ph.interpolate_dynamical_matrices_new(ph['pp'], D + Pi, q, nq,
    apply_asr=True)

info('Plot electrons')

e, U, order = elphmod.dispersion.dispersion(el.H, k, vectors=True, order=True)
e -= mu

if comm.rank == 0:
    proj = 0.1 * abs(U) ** 2

    for n in range(el.size):
        fatbands = elphmod.plot.compline(x, e[:, n], proj[:, :, n])

        for fatband, color in zip(fatbands, 'rgb'):
            plt.fill(*fatband, color=color, linewidth=0.0)

    plt.ylabel(r'$\epsilon$ (eV)')
    plt.xlabel(r'$k$')
    plt.xticks(x[GMKG], 'GMKG')
    plt.show()

info('Plot cDFPT, DFPT and renormalized phonons')

for method in sorted(ph):
    w2, u, order = elphmod.dispersion.dispersion(ph[method].D, k,
        vectors=True, order=True)

    w = elphmod.ph.sgnsqrt(w2) * Ry2eV * 1e3

    if comm.rank == 0:
        proj = elphmod.ph.polarization(u, k)

        for nu in range(ph[method].size):
            fatbands = elphmod.plot.compline(x, w[:, nu], proj[:, nu])

            for fatband, color in zip(fatbands, 'ymk'):
                plt.fill(*fatband, color=color, linewidth=0.0)

        plt.ylabel(r'$\omega$ (meV)')
        plt.xlabel(r'$q$')
        plt.xticks(x[GMKG], 'GMKG')
        plt.show()
