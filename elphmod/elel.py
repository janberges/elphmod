# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

"""Coulomb interaction from VASP."""

import sys
import numpy as np

import elphmod.bravais
import elphmod.dispersion
import elphmod.misc
import elphmod.MPI

comm = elphmod.MPI.comm

class Model:
    r"""Localized model for electron-electron interaction.

    Currently, only square and hexagonal Bravais lattices are supported.

    Parameters
    ----------
    uijkl : str
        File with Coulomb tensor in orbital basis on uniform q mesh.
    vijkl_full, vijkl_redu : str
        Files with full and reduced bare Coulomb tensors. The difference is
        added as a correction to the Coulomb tensor provided via `uijkl`.
    nq : tuple of int or int
        Number of q points per dimension. If an integer is given, a 2D mesh with
        `nq` points in both the first and the second dimension is assumed (for
        backward compatibility). If omitted or ``None``, lattice vectors and
        corresponding interactions from `Wmat` are used unmodified.
    no : int
        Number of orbitals. Ignored if `Wmat` is used.
    Wmat : str
        File with density-density Coulomb matrix in orbitals basis for different
        lattice vectors.
    a : ndarray, optional
        Bravais lattice vectors. By default, a 2D lattice with `angle` between
        the first and the second basis vector is assumed.
    r : ndarray, optional
        Positions of orbital centers. By default, all orbitals are assumed to be
        located at the origin of the unit cell.
    angle : number, default 120
        Angle between lattice vectors in degrees.

    Attributes
    ----------
    R : ndarray
        Lattice vectors :math:`\vec R` of Wigner-Seitz supercell.
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
        r"""Set up density-density Coulomb matrix for arbitrary q point.

        Parameters
        ----------
        q1, q2, q3 : float, default 0.0
            q point in crystal coordinates with period :math:`2 \pi`.

        Returns
        -------
        ndarray
            Fourier transform of :attr:`data`.
        """
        q = np.array([q1, q2, q3])

        return np.einsum('Rab,R->ab', self.data, np.exp(1j * self.R.dot(q)))

    def WR(self, R1=0, R2=0, R3=0):
        """Get density-density Coulomb matrix for arbitrary lattice vector.

        Parameters
        ----------
        R1, R2, R3 : int, default 0
            Lattice vector in units of primitive vectors.

        Returns
        -------
        ndarray
            Element of :attr:`data` or zero.
        """
        index = elphmod.misc.vector_index(self.R, (R1, R2, R3))

        if index is None:
            return np.zeros_like(self.data[0])
        else:
            return self.data[index]

    def __init__(self, uijkl=None, vijkl_full=None, vijkl_redu=None,
            nq=None, no=None, Wmat=None, a=None, r=None, angle=120):

        self.cells = [(0, 0, 0)]

        if uijkl is None and Wmat is None:
            return

        if hasattr(nq, '__len__'):
            nq_orig = nq
            nq = np.ones(3, dtype=int)
            nq[:len(nq_orig)] = nq_orig
        elif nq:
            nq = np.array([nq, nq, 1])

        if a is None:
            a = np.zeros((3, 3))
            a[:2, :2] = elphmod.bravais.translations(angle)
            a[2, 2] = 1.0

        if Wmat is not None:
            R, data = elphmod.misc.read_dat_mat(Wmat)

            self.size = data.shape[1]
        else:
            Wq = read_orbital_Coulomb_interaction(uijkl, nq, no, dd=True)

            if vijkl_full is not None and vijkl_redu is not None:
                Wq += read_orbital_Coulomb_interaction(vijkl_full, nq, no,
                    dd=True)
                Wq -= read_orbital_Coulomb_interaction(vijkl_redu, nq, no,
                    dd=True)

            self.size = no

        if r is None:
            r = np.zeros((self.size, 3))

        if Wmat is not None:
            if nq is None:
                self.R = R
                self.data = data
                return

            WR = np.zeros((*nq, self.size, self.size), dtype=complex)

            for iR, (R1, R2, R3) in enumerate(R):
                WR[R1 % nq[0], R2 % nq[1], R3 % nq[2]] = data[iR]

            q2r(self, WR, a, r, fft=False)
        else:
            q2r(self, Wq, a, r)

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
        elel = Model()

        supercell = elphmod.bravais.supercell(N1, N2, N3)
        elel.cells = supercell[-1]

        elel.size = len(elel.cells) * self.size

        if comm.rank == 0:
            const = dict()

            status = elphmod.misc.StatusBar(len(elel.cells),
                title='map interaction onto supercell')

            for i in range(len(elel.cells)):
                A = i * self.size

                for n in range(len(self.R)):
                    R, r = elphmod.bravais.to_supercell(
                        self.R[n] + elel.cells[i], supercell)

                    B = r * self.size

                    if R not in const:
                        const[R] = np.zeros((elel.size, elel.size),
                            dtype=complex)

                    const[R][A:A + self.size, B:B + self.size] = self.data[n]

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

    def standardize(self, eps=0.0):
        r"""Standardize real-space interaction data.

        - Keep only nonzero matrix elements.
        - Sum over repeated lattice vectors.
        - Sort lattice vectors.

        Parameters
        ----------
        eps : float
            Threshold for "nonzero" matrix elements in units of the maximum
            matrix element.
        """
        if comm.rank == 0:
            if eps:
                self.data[abs(self.data) < eps * abs(self.data).max()] = 0.0

            const = dict()

            status = elphmod.misc.StatusBar(len(self.R),
                title='standardize interaction data')

            for n in range(len(self.R)):
                if np.any(self.data[n] != 0.0):
                    R = tuple(self.R[n])

                    if R in const:
                        const[R] += self.data[n]
                    else:
                        const[R] = self.data[n].copy()

                status.update()

            cells = sorted(list(const.keys()))
            count = len(cells)

            self.R = np.array(cells, dtype=int)
            self.data = np.array([const[R] for R in cells])
        else:
            count = None

        count = comm.bcast(count)

        if comm.rank != 0:
            self.R = np.empty((count, 3), dtype=int)
            self.data = np.empty((count, self.size, self.size), dtype=complex)

        comm.Bcast(self.R)
        comm.Bcast(self.data)

    def to_Wmat(self, Wmat):
        if comm.rank == 0:
            with open(Wmat, 'w') as data:
                data.write('# Coulomb interaction W at omega = 0\n')
                data.write('# 1: R1, 2: R2, 3: R3 (lattice vector)\n')
                data.write('# 1: i, 2: j, 3: Re[W] (eV), 4: Im[W] (eV)\n')

                for n in range(len(self.R)):
                    data.write(('%12d' * 3 + '\n') % tuple(self.R[n]))

                    for a in range(self.size):
                        for b in range(self.size):
                            data.write('%5d%4d%20.10f%20.10f\n' % (a + 1, b + 1,
                                self.data[n, a, b].real,
                                self.data[n, a, b].imag))

                    data.write('\n')

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
        U = np.empty((*nq, no, no), dtype=complex)
    else:
        U = np.empty((*nq, no, no, no, no), dtype=complex)

    if comm.rank == 0:
        with open(filename) as data:
            for line in data:
                try:
                    columns = line.split()

                    q1, q2, q3 = [int(round(float(columns[c]) * nq[c])) % nq[c]
                        for c in range(3)]

                    i, j, k, l = [int(n) - 1
                        for n in columns[3:7]]

                    if not dd:
                        U[q1, q2, q3, j, i, l, k] \
                            = float(columns[7]) + 1j * float(columns[8])
                    elif i == j and k == l:
                        U[q1, q2, q3, i, k] \
                            = float(columns[7]) + 1j * float(columns[8])
                except (ValueError, IndexError):
                    continue

    comm.Bcast(U)

    return U

def q2r(elel, W, a, r, fft=True):
    """Interpolate electron-electron interaction on uniform q-point mesh.

    Parameters
    ----------
    elel : :class:`elphmod.elel.Model`
        Localized model for electron-electron interaction.
    W : ndarray
        Density-density interaction matrices on complete uniform q-point mesh.
    a : ndarray
        Bravais lattice vectors.
    r : ndarray
        Positions of orbital centers.
    fft : bool
        Perform Fourier transform? If ``False``, only the mapping to the
        Wigner-Seitz cell is performed.
    """
    nq = W.shape[:-2]
    elel.size = W.shape[-2]

    nq_orig = tuple(nq)
    nq = np.ones(3, dtype=int)
    nq[:len(nq_orig)] = nq_orig

    if fft:
        Wq = np.reshape(W, (*nq, elel.size, elel.size))
        WR = np.fft.ifftn(Wq.conj(), axes=(0, 1, 2)).conj()
    else:
        WR = W

    WR = np.reshape(WR, (*nq, elel.size, 1, elel.size, 1))
    WR = np.transpose(WR, (3, 5, 0, 1, 2, 4, 6))

    elel.R, elel.data, l = elphmod.bravais.short_range_model(WR, a, r, sgn=+1)

def read_band_Coulomb_interaction(filename, nQ, nk, binary=False, share=False):
    """Read Coulomb interaction for single band in band basis."""

    if share:
        node, images, U = elphmod.MPI.shared_array((nQ, nk, nk, nk, nk),
            dtype=complex)
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

    psi = elphmod.dispersion.dispersion_full_nosym(H, nk, vectors=True,
        gauge=True)[1]
    # psi[k, a, n] = <a k|n k>

    psi = psi[:, :, :, band].copy()

    # distribute work among processors:

    Q = sorted(elphmod.bravais.irreducibles(nq)) if comm.rank == 0 else None
    Q = comm.bcast(Q)
    nQ = len(Q)

    size = nQ * nk ** 4

    sizes = elphmod.MPI.distribute(size)

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

    chunk = elphmod.MPI.MPI.UNSIGNED_CHAR.Create_contiguous(10).Commit()
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
        node, images, V = elphmod.MPI.shared_array((nQ, nk, nk, nk, nk),
            dtype=complex)
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
    charge-density output. Use :func:`elphmod.el.read_rhoG_density` for this
    purpose.

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
        fac = 1 / np.linalg.norm(g_vect[ig]) ** 2
        rgtot_re = rho_g[ig].real
        rgtot_im = rho_g[ig].imag
        ehart = ehart + (rgtot_re ** 2 + rgtot_im ** 2) * fac

    e2 = 2.0
    fpi = 4 * np.pi
    tpiba2 = (2 * np.pi / (a / elphmod.misc.a0)) ** 2

    fac = e2 * fpi / tpiba2
    ehart = ehart * fac

    ehart = ehart * 0.5 * uc_volume

    return ehart
