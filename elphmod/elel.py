#/usr/bin/env python

from . import bravais

import numpy as np

def read_orbital_Coulomb_interaction(comm, filename, nq, no):
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

def read_band_Coulomb_interaction(comm, filename, nQ, nk):
    """Read Coulomb interaction for single band in band basis.."""

    U = np.empty((nQ, nk, nk, nk, nk), dtype=complex)

    if comm.rank == 0:
        with open(filename) as data:
            for iQ in range(len(Q)):
                for k1 in range(nk):
                    for k2 in range(nk):
                        for K1 in range(nk):
                            for K2 in range(nk):
                                ReU, ImU = list(map(float, next(data).split()))
                                U[iQ, k1, k2, K1, K2] = ReU + 1j * ImU

    comm.Bcast(U)

    return U
