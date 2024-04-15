#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

"""
Created on Sun Mar 14 15:15:43 2021

@author: arne
"""

import elphmod
import re
import numpy as np
import sys
import matplotlib.pyplot as plt

# Note: returns angle in degree
def theta(v, w):
    return np.arccos(v.dot(w)
        / (np.linalg.norm(v) * np.linalg.norm(w))) / np.pi * 180

def supercell_vectors(cdw_data, N1, N2, A, a, a1, a2, a3):

    if cdw_data['ibrav'] == 4:

        angle = theta(a1, a2)

        # Lattice vectors of CDW structure:
        A1 = a1 * N1
        A3 = a3

        eps = 1e-5
        for n in range(50):
            for m in range(50):
                test_lattice_vector = n * a1 + m * a2
                if abs(np.linalg.norm(test_lattice_vector) - N2 * a) < eps:
                    if abs(theta(test_lattice_vector, A1) - angle) < eps:
                        A2 = n * a1 + m * a2
        return A1, A2, A3

    elif cdw_data['ibrav'] == 8:

        angle = 90

        if not isinstance(N1, int):
            print('N1 is not an integer')
            eps = 1e-8
            for n in range(-50, 50):
                for m in range(-50, 50):
                    test_lattice_vector = n * a1 + m * a2
                    if abs(np.linalg.norm(test_lattice_vector) - N1 * a) < eps:
                        print(n, m)

                        plt.plot([0, test_lattice_vector[0]],
                            [0, test_lattice_vector[1]],
                            color='black', linewidth=5)

        else:
            # Lattice vectors of CDW structure:
            A1 = N1 * a1
            A3 = a3

        eps = 1e-5
        for n in range(50):
            for m in range(50):
                test_lattice_vector = n * a1 + m * a2
                if abs(np.linalg.norm(test_lattice_vector) - N2 * a) < eps:
                    if abs(theta(test_lattice_vector, A1) - angle) < eps:
                        A2 = n * a1 + m * a2

        return A1, A2, A3
    elif cdw_data['ibrav'] == 0:
        v1 = cdw_data['r_cell'][0, :]
        v2 = cdw_data['r_cell'][1, :]
        v3 = cdw_data['r_cell'][2, :]

        # Calculate angle of supercell vectors
        angle = theta(a1, a2)
        A1 = A * v1
        A2 = A * v2
        A3 = A * v3

        return A1, A2, A3

def permutation_finder(nat, R_cdw, R_sym, at_cdw, at_sym, eps):
    permutation = []
    # Find permutation
    eps = eps
    for atom_index_sym in range(nat):
        for atom_index_cdw in range(nat):
            test_position = R_cdw[atom_index_cdw]
            if np.linalg.norm(test_position - R_sym[atom_index_sym]) < eps:
                permutation.append(atom_index_cdw)

    R_cdw_permuted = np.empty((nat, 3))
    at_cdw_permuted = []
    for index in range(len(permutation)):
        R_cdw_permuted[index] = R_cdw[permutation[index]]
        at_cdw_permuted.append(at_cdw[permutation[index]])

    at_cdw = at_cdw_permuted
    R_cdw = R_cdw_permuted

    return R_cdw, at_cdw

def align_structures(nat, R_cdw, R_sym, A1, A2, eps):
    # Move all atoms in the CDW structure
    # and check if they align with SYM structure
    eps = eps
    for atom_index_sym in range(nat):
        for atom_index_cdw in range(nat):
            for m in [-1, 0, 1]:
                for n in [-1, 0, 1]:
                    test_position = R_cdw[atom_index_cdw] + m * A1 + n * A2
                    if (np.linalg.norm(test_position - R_sym[atom_index_sym])
                            < eps):
                        R_cdw[atom_index_cdw] = test_position

    return R_cdw
