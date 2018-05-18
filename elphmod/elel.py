#/usr/bin/env python

import sys
import numpy as np

from . import bravais, dispersion, MPI
comm = MPI.comm

def read_orbital_Coulomb_interaction(filename, nq, no):
    """Read Coulomb interaction in orbital basis.."""

    U = np.empty((nq, nq, no, no, no, no), dtype=complex)

    if comm.rank == 0:
        with open(filename) as data:
            next(data)
            next(data)

            for line in data:
                columns = line.split()

                q1, q2 = [int(round(float(q) * nq)) % nq for q in columns[0:2]]

                i, j, k, l = [int(n) - 1 for n in columns[3:7]]

                U[q1, q2, j, i, l, k] \
                    = float(columns[7]) + 1j * float(columns[8])

    comm.Bcast(U)

    return U

def read_band_Coulomb_interaction(filename, nQ, nk, binary=False):
    """Read Coulomb interaction for single band in band basis.."""

    U = MPI.shared_array((nQ, nk, nk, nk, nk), dtype=complex)

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
                                    a, b = list(map(float,next(data).split()))
                                    U[iQ, k1, k2, K1, K2] = a + 1j * b

    comm.Barrier()

    return U

def write_band_Coulomb_interaction(filename, U, binary=False):
    """Write Coulomb interaction for single band in band basis.."""

    nQ, nk, nk, nk, nk = U.shape

    if comm.rank == 0:
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

def orbital2band(U, H, nq, nk, band=0, status=False):
    """Transform Coulomb interaction from orbital basis onto single band."""

    nqC, nqC, no, no, no, no = U.shape

    if nqC % nq:
        print("Output q mesh must be subset of input q mesh!")
        return

    # get eigenvectors of Hamiltonian:

    k = np.empty((nk * nk, 2))

    if comm.rank == 0:
        n = 0
        for k1 in range(nk):
            for k2 in range(nk):
                k[n] = k1, k2
                n += 1

        k *= 2 * np.pi / nk

    eps, psi = dispersion.dispersion(H, k, vectors=True, gauge=True)
    # psi[k, a, n] = <a k|n k>

    psi = np.reshape(psi[:, :, band], (nk, nk, no))

    # distribute work among processors:

    Q = sorted(bravais.irreducibles(nq))

    size = len(Q) * nk ** 4

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
                        Kq1 = (K1 - Q1) % nk

                        for K2 in range(nk):
                            Kq2 = (K2 - Q2) % nk

                            points[n] \
                                = q1, q2, k1, k2, K1, K2, kq1, kq2, Kq1, Kq2

                            n += 1
    else:
        points = None

    my_points = np.empty((sizes[comm.rank], 10), dtype=np.uint8)
    comm.Scatterv((points, sizes * 10), my_points)

    # transform from orbital to band basis:

    my_V = np.zeros(sizes[comm.rank], dtype=complex)

    for n, (q1, q2, k1, k2, K1, K2, kq1, kq2, Kq1, Kq2) in enumerate(my_points):
        if status and comm.rank == 0:
            sys.stdout.write('%3.0f%%\r' % (n / len(my_points) * 100))

        for a in range(no):
            for b in range(no):
                for c in range(no):
                    for d in range(no):
                        my_V[n] += (U[q1, q2, a, b, c, d]
                            * psi[K1,  K2,  d].conj()
                            * psi[k1,  k2,  b].conj()
                            * psi[kq1, kq2, a]
                            * psi[Kq1, Kq2, c])

    if status and comm.rank == 0:
        print('Done.')

    V = np.empty((len(Q), nk, nk, nk, nk), dtype=complex)

    comm.Gatherv(my_V, (V, sizes))

    W = MPI.shared_array((len(Q), nk, nk, nk, nk), dtype=complex)

    if comm.rank == 0:
        W[:] = V

    return W
