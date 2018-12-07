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

H = elphmod.el.hamiltonian('TaS2/TaS2_hr.dat')

info("Diagonalize Hamiltonian along G-M-K-G..")

q, x, GMKG = elphmod.bravais.GMKG(120, corner_indices=True)

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

def uniform_grid(size):
    k = np.empty((size * size, 2))

    for n,(k1,k2) in enumerate(itertools.product(range(size),repeat=2)):
        k[n] = k1,k2

    k *= 2 * np.pi / size
    return k


fullsize = 60    
chi_q_full = elphmod.dispersion.dispersion(chi, uniform_grid(fullsize), broadcast=False)
chi_q_full = np.reshape(-chi_q_full,(fullsize,fullsize))    
plt.imshow(chi_q_full,interpolation='none')
plt.colorbar()
plt.show()

#Now do a FT
N, N = chi_q_full.shape

i = np.arange(N)

transform = np.exp(2j * np.pi / N * np.outer(i, i)) / N
chi_r_full = np.dot(np.dot(transform, chi_q_full), transform)

chi_r_full[0,0]=0

# Need to transform this using 120 degree angle
plt.imshow((chi_r_full.real)[0:20,0:20],interpolation='none')
plt.colorbar()
plt.show()

cos120 = np.cos(2*np.pi/3)
sin120 = np.sin(2*np.pi/3)
chi_r_with_coors = np.array([(sin120*ix,jx+cos120*ix,chi_r_full[ix,jx]) for ix,jx in itertools.product(range(15),repeat=2)])

plt.scatter(chi_r_with_coors[:,0],chi_r_with_coors[:,1],c=chi_r_with_coors[:,2],linewidth=0,s=40,marker='h')
plt.axis('equal')
plt.colorbar()
plt.scatter([0,],[0,],c='black',linewidth=0,s=80,marker='o')
plt.show()

chi_r_abs = np.array([ (elphmod.bravais.squared_distance(ix,jx,120),chi_r_full[ix,jx]) for ix,jx in itertools.product(range(10),repeat=2)])

plt.scatter(np.sqrt(chi_r_abs[:,0]),chi_r_abs[:,1])
plt.axvline(np.sqrt(elphmod.bravais.squared_distance(3,0,120)),c='gray')
plt.axvline(np.sqrt(elphmod.bravais.squared_distance(3,3,120)),c='gray')
#plt.axvline(np.sqrt(elphmod.bravais.squared_distance(6,3,120)),c='gray')
plt.axvline(np.sqrt(elphmod.bravais.squared_distance(6,2,120)),c='red')
plt.axvline(np.sqrt(elphmod.bravais.squared_distance(6,4,120)),c='red')
