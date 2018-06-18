#/usr/bin/env python

import numpy as np

from . import bravais, MPI
comm = MPI.comm

def get_q(filename):
    """Get list of irreducible q points."""

    with open(filename) as data:
        return [list(map(float, line.split()[:2])) for line in data]

def coupling(filename, nQ, nb, nk, bands, Q=None, offset=0,
        completion=True, complete_k=False, squeeze=False, status=False):
    """Read and complete electron-phonon matrix elements."""

    if Q is not None:
        nQ = len(Q)
    else:
        Q = np.arange(nQ, dtype=int) + 1

    sizes = MPI.distribute(nQ)

    elph = np.empty((nQ, nb, bands, bands, nk, nk))

    my_elph = np.empty((sizes[comm.rank], nb, bands, bands, nk, nk))
    my_elph[:] = np.nan

    my_Q = np.empty(sizes[comm.rank], dtype=int)
    comm.Scatterv((Q, sizes), my_Q)

    for n, iq in enumerate(my_Q):
        if status:
            print("Read data for q point %d.." % iq)

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
            if status:
                print("Complete data for q point %d.." % iq)

            for nu in range(nb):
                for ibnd in range(bands):
                    for jbnd in range(bands):
                        bravais.complete(my_elph[n, nu, ibnd, jbnd])

    if complete_k and Q is None: # to be improved considerably
        comm.Gatherv(my_elph, (elph, sizes * nb * bands * bands * nk * nk))

        elph_complete = np.empty((nq, nq, nb, bands, bands, nk, nk))

        if comm.rank == 0:
            symmetries = [image for name, image in elphmod.bravais.symmetries(
                np.zeros((nk, nk)), unity=False)]

            scale = nk // nq

            done = bravais.irreducibles(nq)
            q_irr = sorted(done)

            for sym in symmetries:
                for iq, (q1, q2) in enumerate(q_irr):
                    Q1, Q2 = sym[q1 * scale, q2 * scale] // scale

                    if (Q1, Q2) in done:
                        continue

                    done.add((Q1, Q2))

                    for k1 in range(nk):
                        for k2 in range(nk):
                            K1, K2 = sym[k1, k2]

                            elph_complete[Q1, Q2, ..., K1, K2] \
                                = elph_complete[iq, ..., k1, k2]

        comm.Bcast(elph_complete)
        elph = elph_complete
    else:
        comm.Allgatherv(my_elph, (elph, sizes * nb * bands * bands * nk * nk))

    return elph[..., 0, 0, :, :] if bands == 1 and squeeze else elph

def read_EPW_output(epw_out, q, nq, nb, nk, bands=1,
                    eps=1e-4, squeeze=False, status=False, epf=False):
    """Read electron-phonon coupling from EPW output file."""

    elph = np.empty((len(q), nb, bands, bands, nk, nk),
        dtype=complex if epf else float)

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

                    if status:
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

                    for _ in range(bands * bands * nb):
                        columns = next(data).split()

                        ibnd, jbnd, nu = [int(i) - 1 for i in columns[:3]]

                        if epf:
                            elph[iq, nu, ibnd, jbnd, k1, k2] = complex(
                                float(columns[-2]), float(columns[-1]))
                        else:
                            elph[iq, nu, ibnd, jbnd, k1, k2] = float(
                                columns[-1])

        if np.isnan(elph).any():
            print("Warning: EPW output incomplete!")

        if epf:
            elph *= 1e-3 ** 1.5 # meV^(3/2) to eV^(3/2)
        else:
            elph *= 1e-3 # meV to eV

    comm.Bcast(elph)

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
