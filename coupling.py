#/usr/bin/env python

import bravais
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

def read(filename):
    """Read file with Fermi-surface averaged electron-phonon coupling."""

    elph = dict()

    with open(filename) as data:
        for line in data:
            columns = line.split()

            q = tuple(map(int, columns[:2]))
            elph[q] = list(map(float, columns[2:]))

    return elph

def complete(elph, nq, bands):
    """Generate whole Brillouin zone from irreducible q points."""

    elphmat = np.empty((nq, nq, bands))

    for q in elph.keys():
        for Q in bravais.images(*q, nk=nq):
            elphmat[Q] = elph[q]

    return elphmat

def plot(elphmat, points=50):
    """Plot electron-phonong coupling."""

    nq, nq, bands = elphmat.shape

    qxmax = 2 * bravais.U1[0]
    qymax = 2 * bravais.U2[1]

    nqx = int(round(points * qxmax))
    nqy = int(round(points * qymax))

    qx = np.linspace(0.0, qxmax, nqx, endpoint=False)
    qy = np.linspace(0.0, qymax, nqy, endpoint=False)[::-1]

    image = np.empty((bands, nqy, nqx))

    for nu in range(bands):
        elphfun = bravais.Fourier_interpolation(elphmat[:, :, nu])

        for i in range(len(qy)):
            for j in range(len(qx)):
                q1 = qx[j] * bravais.T1[0] + qy[i] * bravais.T1[1]
                q2 = qx[j] * bravais.T2[0] + qy[i] * bravais.T2[1]

                image[nu, i, j] = elphfun(q1 * nq, q2 * nq)

    return \
        np.concatenate([
        np.concatenate(
            image[3 * n:3 * n + 3],
        axis=1) for n in range(bands // 3)],
        axis=0)
