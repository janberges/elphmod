#!/usr/bin/env python3

# Copyright (C) 2017-2022 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import sys
import numpy as np

from . import bravais, dispersion, misc, MPI
comm = MPI.comm

class Model(object):
    """Localized model for electron-electron interaction.

    Currently, only square and hexagonal Bravais lattices are supported.

    Parameters
    ----------
    uijkl : str
        File with Coulomb tensor in orbital basis on uniform q mesh.
    nq : int
        Dimension of q mesh.
    no : int
        Number of orbitals.
    angle : numbe, optional
        Angle between lattice vectors.

    Attributes
    ----------
    R : ndarray
        Lattice vectors of Wigner-Seitz supercell.
    data : ndarray
        Corresponding density-density interaction in orbital basis.
    size : int
        Number of Wannier functions.
    cells : list of tuple of int, optional
        Lattice vectors of unit cells if the model describes a supercell.
    N : list of tuple of int, optional
        Primitive vectors of supercell if the model describes a supercell.
    """
    def W(self, q1=0, q2=0, q3=0):
        """Set up density-density Coulomb matrix for arbitrary q point."""

        q = np.array([q1, q2, q3])

        return np.einsum('Rab,R->ab', self.data, np.exp(-1j * self.R.dot(q)))

    def __init__(self, uijkl=None, nq=None, no=None, angle=120):
        if uijkl is None:
            return

        Wq = read_orbital_Coulomb_interaction(uijkl, nq, no, dd=True)
        Wq = Wq.reshape((nq, nq, 1, no, no))

        WR = np.fft.ifftn(Wq, axes=(0, 1, 2))

        irvec, ndegen, wslen = bravais.wigner_seitz(nq, angle=angle)

        self.R = np.zeros((len(irvec), 3), dtype=int)
        self.data = np.empty((len(irvec), no, no), dtype=complex)

        self.R[:, :2] = irvec

        for i in range(len(self.R)):
            self.data[i] = WR[tuple(self.R[i] % nq)] / ndegen[i]

        self.size = no

    def supercell(self, N1=1, N2=1, N3=1):
        """Map localized model for electron-electron interaction onto supercell.

        Parameters
        ----------
        N1, N2, N3 : tuple of int or int, default 1
            Supercell lattice vectors in units of primitive lattice vectors.
            Multiples of single primitive vector can be defined via a scalar
            integer, linear combinations via a 3-tuple of integers.

        Returns
        -------
        object
            Localized model for electron-electron interaction for supercell.
        """
        if not hasattr(N1, '__len__'): N1 = (N1, 0, 0)
        if not hasattr(N2, '__len__'): N2 = (0, N2, 0)
        if not hasattr(N3, '__len__'): N3 = (0, 0, N3)

        N1 = np.array(N1)
        N2 = np.array(N2)
        N3 = np.array(N3)

        N = np.dot(N1, np.cross(N2, N3))

        B1 = np.cross(N2, N3)
        B2 = np.cross(N3, N1)
        B3 = np.cross(N1, N2)

        elel = Model()
        elel.size = N * self.size
        elel.cells = []
        elel.N = [tuple(N1), tuple(N2), tuple(N3)]

        if comm.rank == 0:
            for n1 in range(N):
                for n2 in range(N):
                    for n3 in range(N):
                        indices = n1 * N1 + n2 * N2 + n3 * N3

                        if np.all(indices % N == 0):
                            elel.cells.append(tuple(indices // N))

            assert len(elel.cells) == N

            const = dict()

            status = misc.StatusBar(len(self.R),
                title='map interaction onto supercell')

            for n in range(len(self.R)):
                for i, cell in enumerate(elel.cells):
                    R = self.R[n] + np.array(cell)

                    R1, r1 = divmod(np.dot(R, B1), N)
                    R2, r2 = divmod(np.dot(R, B2), N)
                    R3, r3 = divmod(np.dot(R, B3), N)

                    R = R1, R2, R3

                    indices = r1 * N1 + r2 * N2 + r3 * N3
                    j = elel.cells.index(tuple(indices // N))

                    A = i * self.size
                    B = j * self.size

                    if R not in const:
                        const[R] = np.zeros((elel.size, elel.size),
                            dtype=complex)

                    const[R][B:B + self.size, A:A + self.size] = self.data[n]

                status.update()

            elel.R = np.array(list(const.keys()), dtype=int)
            elel.data = np.array(list(const.values()))

            count = len(const)
            const.clear()
        else:
            count = None

        count = comm.bcast(count)

        if comm.rank != 0:
            elel.R = np.empty((count, 3), dtype=int)
            elel.data = np.empty((count, elel.size, elel.size), dtype=complex)

        comm.Bcast(elel.R)
        comm.Bcast(elel.data)

        elel.cells = comm.bcast(elel.cells)

        return elel

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

def read_orbital_Coulomb_interaction(filename, nq, no, dd=False):
    """Read Coulomb interaction in orbital basis."""

    if dd:
        U = np.empty((nq, nq, no, no), dtype=complex)
    else:
        U = np.empty((nq, nq, no, no, no, no), dtype=complex)

    if comm.rank == 0:
        with open(filename) as data:
            for line in data:
                try:
                    columns = line.split()

                    q1, q2 = [int(round(float(q) * nq)) % nq
                        for q in columns[0:2]]

                    i, j, k, l = [int(n) - 1
                        for n in columns[3:7]]

                    if not dd:
                        U[q1, q2, j, i, l, k] \
                            = float(columns[7]) + 1j * float(columns[8])
                    elif i == j and k == l:
                        U[q1, q2, i, k] \
                            = float(columns[7]) + 1j * float(columns[8])
                except (ValueError, IndexError):
                    continue

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
