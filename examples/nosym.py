#/usr/bin/env python

import elphmod

import sys
import numpy as np
import matplotlib.pyplot as plt

comm = elphmod.MPI.comm
info = elphmod.MPI.info

def print_matrix_of_complex_vectors(data):
    if comm.rank == 0:
        I, J, K = data.shape

        for i in range(I):
            for j in range(J):
                sys.stdout.write(' [')

                for k in range(K):
                    sys.stdout.write(' %4.1f%+4.1fi'
                        % (data[i, j, k].real, data[i, j, k].imag))

                sys.stdout.write(' ]')

            sys.stdout.write('\n')

nk = 3

info("Set up Wannier Hamiltonian..")

H = elphmod.electrons.hamiltonian('data/NbSe2_hr.dat')

info("Diagonalize Hamiltonian on uniform mesh..")

eps1, psi1 = elphmod.dispersion.dispersion_full_nosym(H, nk,
    vectors=True, gauge=True, rotate=False)

print_matrix_of_complex_vectors(psi1[:, :, :, 0])

info("Diagonalize Hamiltonian on uniform mesh using symmetry..")

eps2, psi2 = elphmod.dispersion.dispersion_full(H, nk,
    vectors=True, gauge=True, rotate=False)

print_matrix_of_complex_vectors(psi2[:, :, :, 0])
