#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import sys
import numpy as np

from . import bravais, dispersion, MPI
comm = MPI.comm

def read_local_Coulomb_tensor(filename, no):
    """Read local Coulomb tensor from VASP."""

    U = np.empty((no, no, no, no), dtype=complex)

    with open(filename) as data:
        for line in data:
            columns = line.split()

            i, j, k, l = [int(n) - 1 for n in columns[:4]]
            Re, Im = [float(n) for n in columns[4:]]

            U[i, j, k, l] = float(Re) + 1j * float(Im)

    return U

def read_orbital_Coulomb_interaction(filename, nq, no, dd=False, skip=2):
    """Read Coulomb interaction in orbital basis."""

    if dd:
        U = np.empty((nq, nq, no, no), dtype=complex)
    else:
        U = np.empty((nq, nq, no, no, no, no), dtype=complex)

    if comm.rank == 0:
        with open(filename) as data:
            for _ in range(skip):
                next(data)

            for line in data:
                columns = line.split()

                q1, q2 = [int(round(float(q) * nq)) % nq for q in columns[0:2]]

                i, j, k, l = [int(n) - 1 for n in columns[3:7]]

                if not dd:
                    U[q1, q2, j, i, l, k] \
                        = float(columns[7]) + 1j * float(columns[8])
                elif i == j and k == l:
                    U[q1, q2, i, k] \
                        = float(columns[7]) + 1j * float(columns[8])

    comm.Bcast(U)

    return U

def read_band_Coulomb_interaction(filename, nQ, nk, binary=False, share=False):
    """Read Coulomb interaction for single band in band basis."""

    if share:
        node, images, U = MPI.shared_array((nQ, nk, nk, nk, nk), dtype=complex)
    else:
        if comm.rank == 0:
            U = np.empty((nQ, nk, nk, nk, nk), dtype=complex)
        else:
            U = None

    if comm.rank == 0:
        if binary:
            if not filename.endswith('.npy'):
                filename += '.npy'

            U[:] = np.load(filename)
        else:
            with open(filename) as data:
                for iQ in range(nQ):
                    for k1 in range(nk):
                        for k2 in range(nk):
                            for K1 in range(nk):
                                for K2 in range(nk):
                                    a, b = list(map(float, next(data).split()))
                                    U[iQ, k1, k2, K1, K2] = a + 1j * b

    if share:
        if node.rank == 0:
            images.Bcast(U)

        comm.Barrier()

    return U

def write_band_Coulomb_interaction(filename, U, binary=False):
    """Write Coulomb interaction for single band in band basis."""

    if comm.rank == 0:
        nQ, nk, nk, nk, nk = U.shape

        if binary:
            np.save(filename, U)
        else:
            with open(filename, 'w') as data:
                for iQ in range(nQ):
                    for k1 in range(nk):
                        for k2 in range(nk):
                            for K1 in range(nk):
                                for K2 in range(nk):
                                    data.write('%14.9f %14.9f\n' % (
                                        U[iQ, k1, k2, K1, K2].real,
                                        U[iQ, k1, k2, K1, K2].imag))

def orbital2band(U, H, nq, nk, band=0, status=False, share=False, dd=False):
    """Transform Coulomb interaction from orbital basis onto single band."""

    nqC, nqC, no, no, no, no = U.shape

    if nqC % nq:
        print('Output q mesh must be subset of input q mesh!')
        return

    # get eigenvectors of Hamiltonian:

    psi = dispersion.dispersion_full_nosym(H, nk, vectors=True, gauge=True)[1]
    # psi[k, a, n] = <a k|n k>

    psi = psi[:, :, :, band].copy()

    # distribute work among processors:

    Q = sorted(bravais.irreducibles(nq)) if comm.rank == 0 else None
    Q = comm.bcast(Q)
    nQ = len(Q)

    size = nQ * nk ** 4

    sizes = MPI.distribute(size)

    if comm.rank == 0:
        points = np.empty((size, 10), dtype=np.uint8)

        n = 0

        for iq, (q1, q2) in enumerate(Q):
            Q1 = q1 * nk // nq
            Q2 = q2 * nk // nq

            q1 *= nqC // nq
            q2 *= nqC // nq

            for k1 in range(nk):
                kq1 = (k1 + Q1) % nk

                for k2 in range(nk):
                    kq2 = (k2 + Q2) % nk

                    for K1 in range(nk):
                        Kq1 = (K1 + Q1) % nk

                        for K2 in range(nk):
                            Kq2 = (K2 + Q2) % nk

                            points[n] \
                                = q1, q2, k1, k2, K1, K2, kq1, kq2, Kq1, Kq2

                            n += 1
    else:
        points = None

    my_points = np.empty((sizes[comm.rank], 10), dtype=np.uint8)

    # Chunk-wise scattering to overcome MPI's array-length limit of 2^32 - 1:
    # (adapted from L. Dalcin's reply to 'Gatherv seg fault?' on Google Groups)

    chunk = MPI.MPI.UNSIGNED_CHAR.Create_contiguous(10).Commit()
    comm.Scatterv((points, sizes, chunk), (my_points, chunk))
    chunk.Free()

    # transform from orbital to band basis:
    #
    #  ---<---b           c---<---
    #     k    \    q    /    K
    #           o~~~~~~~o
    #    k+q   /         \   K+q
    #  --->---a           d--->---

    my_V = np.zeros(sizes[comm.rank], dtype=complex)

    for n, (q1, q2, k1, k2, K1, K2, kq1, kq2, Kq1, Kq2) in enumerate(my_points):
        if status and comm.rank == 0:
            sys.stdout.write('%3.0f%%\r' % (n / len(my_points) * 100))
            sys.stdout.flush()

        if dd: # consider only density-density terms
            for a in range(no):
                for b in range(no):
                    my_V[n] += (U[q1, q2, a, a, b, b]
                        * psi[Kq1, Kq2, b].conj()
                        * psi[k1, k2, a].conj()
                        * psi[kq1, kq2, a]
                        * psi[K1, K2, b])
        else:
            for a in range(no):
                for b in range(no):
                    for c in range(no):
                        for d in range(no):
                            my_V[n] += (U[q1, q2, a, b, c, d]
                                * psi[Kq1, Kq2, d].conj()
                                * psi[k1, k2, b].conj()
                                * psi[kq1, kq2, a]
                                * psi[K1, K2, c])

    if status and comm.rank == 0:
        print('Done.')

    if share:
        node, images, V = MPI.shared_array((nQ, nk, nk, nk, nk), dtype=complex)
    else:
        if comm.rank == 0:
            V = np.empty((nQ, nk, nk, nk, nk), dtype=complex)
        else:
            V = None

    comm.Gatherv(my_V, (V, sizes))

    if share:
        if node.rank == 0:
            images.Bcast(V)

        comm.Barrier()

    return V
