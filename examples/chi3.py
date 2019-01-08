# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 17:44:21 2018

@author: evloon
"""

#/usr/bin/env python

import elphmod

import numpy as np
import matplotlib.pyplot as plt

import itertools

comm = elphmod.MPI.comm
info = elphmod.MPI.info

from_diagrams=True

eF = -0.5486

temperature = 5.0

info("Set up Wannier Hamiltonian..")

H = elphmod.el.hamiltonian('data/NbSe2_hr.dat')

info("Diagonalize Hamiltonian along G-M-K-G..")

q, x, GMKG = elphmod.bravais.GMKG(12, corner_indices=True)

eps, psi, order = elphmod.dispersion.dispersion(H, q, vectors=True, order=True)

eps -= eF

info("Diagonalize Hamiltonian on uniform mesh..")

nk = 60

eps_full = elphmod.dispersion.dispersion_full(H, nk) - eF

info("Calculate DOS of metallic band..")

ne = 300

e = np.linspace(eps_full[:, :, 0].min(), eps_full[:, :, 0].max(), ne)

DOS = elphmod.dos.hexDOS(eps_full[:, :, 0])(e)

info("Plot dispersion and DOS..")

if comm.rank == 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    ax1.set_ylabel('energy (eV)')
    ax1.set_xlabel('wave vector')
    ax2.set_xlabel('density of states (1/eV)')

    ax1.set_xticks(x[GMKG])
    ax1.set_xticklabels('GMKG')

    for i in range(H.size):
        X, Y = elphmod.plot.compline(x, eps[:, i],
            0.05 * (psi[:, :, i] * psi[:, :, i].conj()).real)

        for j in range(H.size):
            ax1.fill(X, Y[j], color='RCB'[j])

    ax2.fill(DOS, e, color='C')

    plt.show()

info("Interpolate dispersion onto very dense k mesh..")

eps_dense = elphmod.bravais.resize(eps_full[:, :, 0], shape=(120, 120))

info("Calculate electron susceptibility along G-M-K-G..")

chi = elphmod.diagrams.susceptibility(eps_dense,T=temperature)
chi_q = elphmod.dispersion.dispersion(chi, q[1:-1], broadcast=False)

if from_diagrams:
    chi_from_diagrams   = elphmod.diagrams.susceptibility2(eps_dense,T=temperature,hyb_height=0.1)
    chi_from_diagrams_q = elphmod.dispersion.dispersion(chi_from_diagrams, q[1:-1], broadcast=False)

if comm.rank == 0:
    plt.xlabel('wave vector')
    plt.ylabel('susceptibility (1/eV)')
    plt.xticks(x[GMKG], 'GMKG')

    plt.plot(x[1:-1], chi_q)
    plt.plot(x[1:-1], chi_from_diagrams_q,c='red')
    plt.ylim(ymax=0)
    plt.show()
