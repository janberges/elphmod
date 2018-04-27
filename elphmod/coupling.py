#/usr/bin/env python

from . import bravais
import numpy as np

def get_q(filename):
    """Get list of irreducible q points."""

    with open(filename) as data:
        return [list(map(float, line.split()[:2])) for line in data]

def coupling(comm, filename, nQ, nb, nk, bands,
             offset=0, completion=True, squeeze=False):
    """Read and complete electron-phonon matrix elements."""

    sizes = np.empty(comm.size, dtype=int)

    if comm.rank == 0:
        sizes[:] = nQ // comm.size
        sizes[:nQ % comm.size] += 1

    comm.Bcast(sizes)

    elph = np.empty((nQ, nb, bands, bands, nk, nk))

    my_elph = np.empty((sizes[comm.rank], nb, bands, bands, nk, nk))
    my_elph[:] = np.nan

    my_Q = np.empty(sizes[comm.rank], dtype=int)
    comm.Scatterv((np.arange(nQ, dtype=int) + 1, sizes), my_Q)

    for n, iq in enumerate(my_Q):
        with open(filename % iq) as data:
            for line in data:
                columns = line.split()

                if columns[0].startswith('#'):
                    continue

                k1, k2, k3, wk, ibnd, jbnd, nu \
                    = [int(i) - 1 for i in columns[:7]]

                my_elph[n, nu, ibnd - offset, jbnd - offset, k1, k2] = float(
                    columns[7])

    if completion:
        for n in range(sizes[comm.rank]):
            for nu in range(nb):
                for ibnd in range(bands):
                    for jbnd in range(bands):
                        bravais.complete(my_elph[n, nu, ibnd, jbnd])

    comm.Allgatherv(my_elph, (elph, sizes * nb * bands * bands * nk * nk))

    return elph[:, :, 0, 0, :, :] if bands == 1 and squeeze else elph

def read(filename, nq, bands):
    """Read and complete Fermi-surface averaged electron-phonon coupling."""

    elph = np.empty((nq, nq, bands))

    with open(filename) as data:
        for line in data:
            columns = line.split()

            q1 = int(columns[0])
            q2 = int(columns[1])

            for Q1, Q2 in bravais.images(q1, q2, nq):
                for band in range(bands):
                    elph[Q1, Q2, band] = float(columns[2 + band])

    return elph
