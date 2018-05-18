#/usr/bin/env python

import numpy as np

from . import bravais, MPI
comm = MPI.comm

def get_q(filename):
    """Get list of irreducible q points."""

    with open(filename) as data:
        return [list(map(float, line.split()[:2])) for line in data]

def coupling(filename, nQ, nb, nk, bands,
             offset=0, completion=True, squeeze=False):
    """Read and complete electron-phonon matrix elements."""

    sizes = MPI.distribute(nQ)

    elph = np.empty((nQ, nb, bands, bands, nk, nk))

    my_elph = np.empty((sizes[comm.rank], nb, bands, bands, nk, nk))
    my_elph[:] = np.nan

    my_Q = np.empty(sizes[comm.rank], dtype=int)
    comm.Scatterv((np.arange(nQ, dtype=int) + 1, sizes), my_Q)

    for n, iq in enumerate(my_Q):
        print("Read data for q point %d.." % (iq + 1))

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
        for n, iq in enumerate(my_Q):
            print("Complete data for q point %d.." % (iq + 1))

            for nu in range(nb):
                for ibnd in range(bands):
                    for jbnd in range(bands):
                        bravais.complete(my_elph[n, nu, ibnd, jbnd])

    comm.Allgatherv(my_elph, (elph, sizes * nb * bands * bands * nk * nk))

    return elph[:, :, 0, 0, :, :] if bands == 1 and squeeze else elph

def read_EPW_output(epw_out, q, nq, nb, nk, eps=1e-4):
    """Read electron-phonon coupling from EPW output file."""

    elph = np.empty((len(q), nb, nk, nk), dtype=complex)

    if comm.rank == 0:
        q_set = set(q)

        elph[:] = np.nan

        iq = None

        with open(epw_out) as data:
            for line in data:
                if line.startswith('     iq = '):
                    if not q_set:
                        break

                    iq = None

                    columns = line.split()

                    q1f = float(columns[-3]) * nq
                    q2f = float(columns[-2]) * nq

                    q1 = int(round(q1f))
                    q2 = int(round(q2f))

                    if abs(q1f - q1) > eps or abs(q2f - q2) > eps: # q in mesh?
                        continue

                    q1 %= nq
                    q2 %= nq

                    if not (q1, q2) in q_set: # q among chosen irred. points?
                        continue

                    iq = q.index((q1, q2))
                    q_set.remove((q1, q2))

                    print('q = (%d, %d)' % (q1, q2))

                if iq is not None and line.startswith('     ik = '):
                    columns = line.split()

                    k1f = float(columns[-3]) * nk
                    k2f = float(columns[-2]) * nk

                    k1 = int(round(k1f))
                    k2 = int(round(k2f))

                    if abs(k1f - k1) > eps or abs(k2f - k2) > eps: # k in mesh?
                        continue

                    k1 %= nk
                    k2 %= nk

                    next(data)
                    next(data)

                    for nu in range(nb):
                        columns = next(data).split()

                        elph[iq, nu, k1, k2] = complex(
                            float(columns[-2]), float(columns[-1]))

        if np.isnan(elph).any():
            print("Warning: EPW output incomplete!")

        elph *= 1e-3 ** 1.5 # meV^(3/2) to eV^(3/2)

    comm.Bcast(elph)

    return elph

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
