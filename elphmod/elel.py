# Copyright (C) 2017-2023 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

"""Coulomb interaction from VASP."""

from __future__ import division

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
    vijkl_full, vijkl_redu : str
        Files with full and reduced bare Coulomb tensors. The difference is
        added as a correction to the Coulomb tensor provided via `uijkl`.
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

    def WR(self, R1=0, R2=0, R3=0):
        """Get density-density Coulomb matrix for arbitrary lattice vector."""

        index = misc.vector_index(self.R, (R1, R2, R3))

        if index is None:
            return np.zeros_like(self.data[0])
        else:
            return self.data[index]

    def __init__(self, uijkl=None, vijkl_full=None, vijkl_redu=None,
            nq=None, no=None, Wmat=None, angle=120):

        if Wmat is not None:
            R, Wmat = read_Wmat(Wmat, num_wann=no)

            WR = np.zeros((nq, nq, 1, no, no), dtype=complex)

            for iR, (R1, R2, R3) in enumerate(R):
                WR[R1 % nq, R2 % nq, 0] = Wmat[iR]
        else:
            if uijkl is None:
                return

            Wq = read_orbital_Coulomb_interaction(uijkl, nq, no, dd=True)

            if vijkl_full is not None and vijkl_redu is not None:
                Wq += read_orbital_Coulomb_interaction(vijkl_full, nq, no,
                    dd=True)
                Wq -= read_orbital_Coulomb_interaction(vijkl_redu, nq, no,
                    dd=True)

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

def read_local_Coulomb_tensor(filename, no, dd=False):
    """Read local Coulomb tensor from VASP."""

    if dd:
        U = np.empty((no, no), dtype=complex)
    else:
        U = np.empty((no, no, no, no), dtype=complex)

    if comm.rank == 0:
        with open(filename) as data:
            for line in data:
                try:
                    columns = line.split()

                    i, j, k, l = [int(n) - 1 for n in columns[:4]]
                    Re, Im = [float(n) for n in columns[4:]]

                    if not dd:
                        U[i, j, k, l] = float(Re) + 1j * float(Im)
                    elif i == j and k == l:
                        U[i, k] = float(Re) + 1j * float(Im)
                except (ValueError, IndexError):
                    continue

    comm.Bcast(U)

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

def hartree_energy(rho_g, g_vect, ngm_g, uc_volume, a=1.0):
    r"""Calculate the Hartree energy in units of Rydberg.

    This function is basically a copy of Quantum ESPRESSO's subroutine ``v_h``
    from ``v_of_rho.f90``.

    All input parameters can be obtained from reading the formatted
    charge-density output. Use :func:`el.read_rhoG_density` for this purpose.

    Parameters
    ----------
    rho_g : ndarray
        Electronic charge density :math:`\rho(\vec G)` on reciprocal-space grid
        points.
    g_vect : ndarray
        Reciprocal lattice vectors :math:`\vec G`.
    ngm_g : integer
        Number of reciprocal lattice vectors.
    uc_volume: float
        Unit-cell volume.

    Returns
    -------
    ehart : float
        Hartree energy in units of Rydberg.
    """
    ehart = 0.0
    for ig in range(1, ngm_g):
        fac = 1.0 / np.linalg.norm(g_vect[ig]) ** 2
        rgtot_re = rho_g[ig].real
        rgtot_im = rho_g[ig].imag
        ehart = ehart + (rgtot_re ** 2 + rgtot_im ** 2) * fac

    e2 = 2.0
    fpi = 4 * np.pi
    tpiba2 = (2 * np.pi / (a / misc.a0)) ** 2

    fac = e2 * fpi / tpiba2
    ehart = ehart * fac

    ehart = ehart * 0.5 * uc_volume

    return ehart

def read_Wmat(filename, num_wann):
    r"""Read Coulomb matrix elements from "dat.Wmat" (RESPACK)

    Parameters
    ----------
    filename : str
        Name of the file.
    num_wann : integer
        Number of Wannier orbitals.

    Returns
    -------
    ndarray
        Lattice vectors.
    ndarray
        Direct (screened) Coulomb matrix elements.
    """
    respack_file = open(filename)
    lines = respack_file.readlines()
    respack_file.close()

    block = 1 + num_wann ** 2 + 1
    # nR: number of lattice vectors R
    nR = int((len(lines) - 3) / block)
    R = np.empty((nR, 3), dtype=int)
    Rcount = 0

    # allocate W matrix
    W = np.empty((nR, num_wann, num_wann), dtype=complex)
    for line in range(3, len(lines)):
        # read lattice vectors R
        if len(lines[line].split()) == 3:
            R1, R2, R3 = lines[line].split()
            R[Rcount][0] = int(R1)
            R[Rcount][1] = int(R2)
            R[Rcount][2] = int(R3)
        # read matrix elements
        if len(lines[line].split()) == 4:
            n, m, Wreal, Wimag = lines[line].split()
            n = int(n) - 1
            m = int(m) - 1
            Wreal = float(Wreal)
            Wimag = float(Wimag)

            W[Rcount][n, m] = Wreal + 1j * Wimag
        if len(lines[line].split()) == 0:
            Rcount += 1

    return R, W
