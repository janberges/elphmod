#/usr/bin/env python

from . import bravais

import numpy as np

def read_Coulomb_interaction(comm, filename, nQ, nk):
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
