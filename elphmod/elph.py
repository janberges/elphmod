#!/usr/bin/env python3

# Copyright (C) 2017-2022 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

# Some routines in this file follow wan2bloch.f90 of EPW v5.3.1.
# Copyright (C) 2010-2016 S. Ponce', R. Margine, C. Verdi, F. Giustino

import numpy as np

from . import bravais, misc, MPI
comm = MPI.comm

class Model(object):
    """Localized model for electron-phonon coupling.

    Parameters
    ----------
    epmatwp : str
        File with electron-phonon coupling in localized bases from EPW.
    wigner : str
        File with definition of Wigner-Seitz supercells from modified EPW.
    el : object
        Tight-binding model for the electrons.
    ph : object
        Mass-spring model for the phonons.
    old_ws : bool
        Use previous definition of Wigner-Seitz cells?
    divide_mass : bool
        Divide electron-phonon coupling by square root of atomic masses?
    divide_ndegen : bool
        Divide real-space coupling by degeneracy of Wigner-Seitz point? Only
        ``True`` yields correct couplings. ``False`` should only be used for
        debugging.
    shared_memory : bool
        Read coupling from EPW into shared memory?

    Attributes
    ----------
    el, ph : object
        Tight-binding and mass-spring models.
    Rk, Rg : ndarray
        Lattice vectors of Wigner-Seitz supercells.
    dk, dg : ndarray
        Degeneracies of Wigner-Seitz points.
    data : ndarray
        Corresponding electron-phonon matrix elements.
    node, images : MPI.Intracomm
        Communicators between processes that share memory or same ``node.rank``
        if `shared_memory`.
    q : ndarray
        Previously sampled q point, if any.
    gq : ndarray
        Rk-dependent coupling for above q point for possible reuse.
    cells : list of tuple of int, optional
        Lattice vectors of unit cells if the model describes a supercell.
    Rk0 : int
        Index of electronic lattice vector at origin if `ph.lr`.
    """
    def g(self, q1=0, q2=0, q3=0, k1=0, k2=0, k3=0, elbnd=False, phbnd=False,
            broadcast=True, comm=comm):
        r"""Calculate electron-phonon coupling for arbitary points k and k + q.

        .. math::

            \sqrt{2 \omega} g_{\nu m n} = \frac \hbar {\sqrt M}
                \bra{\vec k + \vec q m}
                    \frac{\partial V}{\partial u_\nu}
                \ket{\vec k n}

        Parameters
        ----------
        q1, q2, q2 : float
            q point in crystal coordinates with period :math:`2 \pi`.
        k1, k2, k3 : float
            Ingoing k point in crystal coordinates with period :math:`2 \pi`.
        elbnd : bool
            Transform to electronic band basis? Provided for convenience. Since
            the Hamiltonian is diagonalized on the fly for each requested matrix
            element, this option lacks efficiency and control of complex phases.
            Consider the method `sample` of this class instead.
        phbnd : bool
            Transform to phononic band basis? Provided for convenience. Since
            the dynamical matrix is diagonalized on the fly for each requested
            matrix element, this option lacks efficiency and control of complex
            phases. Consider the method `sample` of this class instead.
        broadcast : bool
            Broadcast result to all processors? If ``False``, returns ``None``
            on all but the first processor.
        comm : MPI communicator
            Group of processors running this function (for parallelization of
            Fourier transforms).

        Returns
        -------
        ndarray
            Electron-phonon matrix element :math:`\sqrt{2 \omega} g_{\nu m n}`
            in Ry\ :sup:`3/2`.
        """
        nRq, nph, nRk, nel, nel = self.data.shape

        q = np.array([q1, q2, q3])
        k = np.array([k1, k2, k3])

        if comm.allreduce(self.q is None or np.any(q != self.q)):
            self.q = q

            Rl, Ru = MPI.distribute(nRq, bounds=True,
                comm=comm)[1][comm.rank:comm.rank + 2]

            my_g = np.einsum('Rxrab,R->xrab',
                self.data[Rl:Ru], np.exp(1j * self.Rg[Rl:Ru].dot(q)))

            # Sign convention in bloch2wan.f90 of EPW:
            # 1222  cfac = EXP(-ci * rdotk) / DBLE(nq)
            # 1223  epmatwp(:, :, :, :, ir) = epmatwp(:, :, :, :, ir)
            #           + cfac * epmatwe(:, :, :, :, iq)

            comm.Allreduce(my_g, self.gq)

            if self.ph.lr and np.any(q != 0):
                g_lr = self.g_lr(q1, q2, q3)

                for a in range(self.el.size):
                    self.gq[:, self.Rk0, a, a] += g_lr

        Rl, Ru = MPI.distribute(nRk, bounds=True,
            comm=comm)[1][comm.rank:comm.rank + 2]

        my_g = np.einsum('xRab,R->xab',
            self.gq[:, Rl:Ru], np.exp(1j * self.Rk[Rl:Ru].dot(k)))

        # Sign convention in bloch2wan.f90 of EPW:
        # 1120  cfac = EXP(-ci * rdotk) / DBLE(nkstot)
        # 1121  epmatw( :, :, ir) = epmatw( :, :, ir)
        #           + cfac * epmats(:, :, ik)

        if broadcast or comm.rank == 0:
            g = np.empty((nph, nel, nel), dtype=complex)
        else:
            g = None

        comm.Reduce(my_g, g)

        if comm.rank == 0:
            # Eigenvector convention in wan2bloch.f90 of EPW:
            # 164  cz(:, :) = chf(:, :)
            # 165  CALL ZHEEVD('V', 'L', nbnd, cz, nbnd, w, ...
            # ...
            # 278  cuf = CONJG(TRANSPOSE(cz))

            # Index convention in wan2bloch.f90 of EPW:
            # 2098  CALL ZGEMM('n', 'n', nbnd, nbnd, nbnd, cone, cufkq, &
            # 2099       nbnd, epmatf(:, :, imode), nbnd, czero, eptmp, nbnd)
            # 2100  CALL ZGEMM('n', 'c', nbnd, nbnd, nbnd, cone, eptmp, &
            # 2101       nbnd, cufkk, nbnd, czero, epmatf(:, :, imode), nbnd)

            if elbnd:
                Uk = np.linalg.eigh(self.el.H(*k))[1]
                Ukq = np.linalg.eigh(self.el.H(*k + q))[1]

                g = np.einsum('am,xab,bn->xmn', Ukq.conj(), g, Uk)

            if phbnd:
                uq = np.linalg.eigh(self.ph.D(*q))[1]

                g = np.einsum('xab,xu->uab', g, uq)

        if broadcast:
            comm.Bcast(g)

        return g

    def g_lr(self, q1=0, q2=0, q3=0):
        """Calculate long-range part of electron-phonon coupling."""

        gq = np.zeros(self.ph.size, dtype=complex)

        for factor, d, q in self.ph.generate_long_range(q1, q2, q3):
            gq += 1j * factor * (d if self.ph.Q is None else d + q).conj()

        return gq

    def gR(self, Rq1=0, Rq2=0, Rq3=0, Rk1=0, Rk2=0, Rk3=0):
        """Get electron-phonon matrix elements for arbitrary lattice vectors."""

        index_q = misc.vector_index(self.Rg, (Rq1, Rq2, Rq3))
        index_k = misc.vector_index(self.Rk, (Rk1, Rk2, Rk3))

        if index_q is None or index_k is None:
            return np.zeros_like(self.data[0, :, 0, :, :])
        else:
            return self.data[index_q, :, index_k, :, :]

    def __init__(self, epmatwp=None, wigner=None, el=None, ph=None,
            old_ws=False, divide_mass=True, divide_ndegen=True,
            shared_memory=False):

        self.el = el
        self.ph = ph
        self.q = None

        if epmatwp is None:
            return

        # read lattice vectors within Wigner-Seitz cell:

        self.Rk, self.dk, self.Rg, self.dg = bravais.read_wigner_file(wigner,
            old_ws=old_ws, nat=ph.nat)

        # read coupling in Wannier basis from EPW output:
        # ('epmatwp' allocated and printed in 'ephwann_shuffle.f90')

        shape = len(self.Rg), ph.size, len(self.Rk), el.size, el.size

        self.node, self.images, self.data = MPI.shared_array(shape,
            dtype=np.complex128, shared_memory=shared_memory)

        if comm.rank == 0:
            with open(epmatwp) as data:
                for irg in range(shape[0]):
                    tmp = np.fromfile(data, dtype=np.complex128,
                        count=np.prod(shape[1:])).reshape(shape[1:])

                    self.data[irg] = np.swapaxes(tmp, 2, 3)

                    # index orders:
                    # EPW (Fortran): a, b, R', x, R
                    # after read-in: R, x, R', b, a
                    # after transp.: R, x, R', a, b

                if np.fromfile(data).size:
                    print('Warning: File "%s" larger than expected!' % epmatwp)

            # undo supercell double counting:

            if divide_ndegen:
                self.divide_ndegen(self.data)

            # divide by square root of atomic masses:

            if self.ph.lr and self.ph.divide_mass != divide_mass:
                print("Warning: Inconsistent 'divide_mass' in 'ph' and 'elph'!")
                print("The value from 'ph' (%r) is used." % self.ph.divide_mass)
                divide_mass = self.ph.divide_mass

            if divide_mass:
                for x in range(ph.size):
                    self.data[:, x] /= np.sqrt(ph.M[x // 3])

        if self.node.rank == 0:
            self.images.Bcast(self.data.view(dtype=float))

        comm.Barrier()

        self.gq = np.empty((ph.size, len(self.Rk), el.size, el.size),
            dtype=complex)

        if self.ph.lr:
            self.Rk0 = misc.vector_index(self.Rk, (0, 0, 0))

    def divide_ndegen(self, g):
        """Divide real-space coupling by degeneracy of Wigner-Seitz point.

        Parameters
        ----------
        g : ndarray
            Real-space coupling.
        """
        for irk in range(len(self.Rk)):
            for m in range(self.el.size):
                M = m if self.dk.shape[1] > 1 else 0
                for n in range(self.el.size):
                    N = n if self.dk.shape[0] > 1 else 0
                    if self.dk[N, M, irk]:
                        g[:, :, irk, m, n] /= self.dk[N, M, irk]
                    else:
                        g[:, :, irk, m, n] = 0.0

        for irg in range(len(self.Rg)):
            for x in range(self.ph.size):
                X = x // 3 if self.dg.shape[2] > 1 else 0
                for m in range(self.el.size):
                    M = m if self.dg.shape[1] > 1 else 0
                    for n in range(self.el.size):
                        N = n if self.dg.shape[0] > 1 else 0
                        if self.dg[N, M, X, irg]:
                            g[irg, x, :, m, n] /= self.dg[N, M, X, irg]
                        else:
                            g[irg, x, :, m, n] = 0.0

    def symmetrize(self):
        r"""Symmetrize electron-phonon coupling.

        .. math::

            g_{\vec q, \vec k} = g_{-\vec q, \vec k + \vec q}^\dagger,
            g_{\vec R, \vec R'} = g_{\vec R - \vec R', -\vec R'}^\dagger
        """
        if comm.rank == 0:
            status = misc.StatusBar(len(self.Rg) * len(self.Rk),
                title='symmetrize coupling')

            for g in range(len(self.Rg)):
                for k in range(len(self.Rk)):
                    G = misc.vector_index(self.Rg, self.Rg[g] - self.Rk[k])
                    K = misc.vector_index(self.Rk, -self.Rk[k])

                    if G is None or K is None:
                        self.data[g, :, k] = 0.0
                    else:
                        self.data[g, :, k] += self.data[
                            G, :, K].swapaxes(1, 2).conj()
                        self.data[g, :, k] /= 2
                        self.data[G, :, K] = self.data[
                            g, :, k].swapaxes(1, 2).conj()

                    status.update()

        if self.node.rank == 0:
            self.images.Bcast(self.data.view(dtype=float))

        comm.Barrier()

    def sample(self, *args, **kwargs):
        """Sample coupling.

        See also
        --------
        sample
        """
        return sample(g=self.g, *args, **kwargs)

    def supercell(self, N1=1, N2=1, N3=1, shared_memory=False):
        """Map localized model for electron-phonon coupling onto supercell.

        Parameters
        ----------
        N1, N2, N3 : tuple of int or int, default 1
            Supercell lattice vectors in units of primitive lattice vectors.
            Multiples of single primitive vector can be defined via a scalar
            integer, linear combinations via a 3-tuple of integers.
        shared_memory : bool, default False
            Store mapped coupling in shared memory?

        Returns
        -------
        object
            Localized model for electron-phonon coupling for supercell.
        """
        if not hasattr(N1, '__len__'): N1 = (N1, 0, 0)
        if not hasattr(N2, '__len__'): N2 = (0, N2, 0)
        if not hasattr(N3, '__len__'): N3 = (0, 0, N3)

        N1 = np.array(N1)
        N2 = np.array(N2)
        N3 = np.array(N3)

        N = np.dot(N1, np.cross(N2, N3))

        B1 = np.sign(N) * np.cross(N2, N3)
        B2 = np.sign(N) * np.cross(N3, N1)
        B3 = np.sign(N) * np.cross(N1, N2)

        N = abs(N)

        elph = Model(
            el=self.el.supercell(N1, N2, N3),
            ph=self.ph.supercell(N1, N2, N3))

        elph.cells = elph.el.cells

        Rg = set()
        Rk = set()
        const = dict()

        counter = 0 # counter for parallelization

        status = misc.StatusBar(len(self.Rg) * len(self.Rk),
            title='map coupling onto supercell (1/2)')

        for g in range(len(self.Rg)):
            for k in range(len(self.Rk)):
                for i, cell in enumerate(elph.cells):
                    counter += 1

                    if counter % comm.size != comm.rank:
                        continue

                    G = self.Rg[g] + np.array(cell)
                    K = self.Rk[k] + np.array(cell)

                    G1, g1 = divmod(np.dot(G, B1), N)
                    G2, g2 = divmod(np.dot(G, B2), N)
                    G3, g3 = divmod(np.dot(G, B3), N)
                    K1, k1 = divmod(np.dot(K, B1), N)
                    K2, k2 = divmod(np.dot(K, B2), N)
                    K3, k3 = divmod(np.dot(K, B3), N)

                    Rg.add((G1, G2, G3))
                    Rk.add((K1, K2, K3))

                    R = G1, G2, G3, K1, K2, K3

                    indices = g1 * N1 + g2 * N2 + g3 * N3
                    n = elph.cells.index(tuple(indices // N))
                    indices = k1 * N1 + k2 * N2 + k3 * N3
                    j = elph.cells.index(tuple(indices // N))

                    A = n * self.ph.size
                    B = i * self.el.size
                    C = j * self.el.size

                    if R not in const:
                        const[R] = np.zeros((elph.ph.size,
                            elph.el.size, elph.el.size), dtype=complex)

                    const[R][
                        A:A + self.ph.size,
                        B:B + self.el.size,
                        C:C + self.el.size] = self.data[g, :, k]

                status.update()

        elph.Rg = np.array(sorted(set().union(*comm.allgather(Rg))))
        elph.Rk = np.array(sorted(set().union(*comm.allgather(Rk))))

        elph.node, self.images, elph.data = MPI.shared_array((len(elph.Rg),
            elph.ph.size, len(elph.Rk), elph.el.size, elph.el.size),
            dtype=complex, shared_memory=shared_memory)

        elph.gq = np.empty((elph.ph.size, len(elph.Rk),
            elph.el.size, elph.el.size), dtype=complex)

        if elph.node.rank == 0:
            elph.data[...] = 0.0

        status = misc.StatusBar(len(elph.Rg) * len(elph.Rk),
            title='map coupling onto supercell (2/2)')

        for g, (G1, G2, G3) in enumerate(elph.Rg):
            for k, (K1, K2, K3) in enumerate(elph.Rk):
                R = G1, G2, G3, K1, K2, K3

                for rank in range(elph.node.size):
                    if elph.node.rank == rank:
                        if R in const:
                            elph.data[g, :, k] += const[R]

                    elph.node.Barrier()

                status.update()

        if elph.node.rank == 0:
            self.images.Allreduce(MPI.MPI.IN_PLACE, elph.data.view(dtype=float))

        return elph

def sample(g, q, nk=None, U=None, u=None, broadcast=True, shared_memory=False):
    """Sample coupling for given q and k points and transform to band basis.

    One purpose of this routine is full control of the complex phase.

    Parameters
    ----------
    g : function
        Electron-phonon coupling in the basis of electronic orbitals and
        Cartesian ionic displacements as a function of q and k in crystal
        coordinates with period :math:`2 \pi`.
    q : list of tuple
        List of q points in crystal coordinates :math:`q_i \in [0, 2 \pi)`.
    nk : int or tuple of int
        Number of k points per dimension, i.e., size of uniform mesh. Different
        numbers of k points along different axes can be specified via a tuple.
        Alternatively, `nk` is inferred from the shape of `U`.
    U : ndarray, optional
        Electron eigenvectors for given k mesh.
        If present, transform from orbital to band basis.
    u : ndarray, optional
        Phonon eigenvectors for given q points.
        If present, transform from displacement to band basis.
    broadcast : bool
        Broadcast result from rank 0 to all processes?
    shared_memory : bool, optional
        Store transformed coupling in shared memory?
    """
    sizes, bounds = MPI.distribute(len(q), bounds=True)
    col, row = MPI.matrix(len(q))

    nph, nel, nel = g().shape

    if nk is None:
        nk = U.shape[:-2]
    elif not hasattr(nk, '__len__'):
        nk = (nk,) * len(q[0])

    nk_orig = tuple(nk)
    nk = np.ones(3, dtype=int)
    nk[:len(nk_orig)] = nk_orig

    q_orig = q
    q = np.zeros((len(q_orig), 3))
    q[:, :len(q_orig[0])] = q_orig

    if U is not None:
        nel = U.shape[-1]
        U = np.reshape(U, (nk[0], nk[1], nk[2], -1, nel))

    if u is not None:
        nph = u.shape[-1]

    my_g = np.empty((sizes[row.rank], nph, nk[0], nk[1], nk[2], nel, nel),
        dtype=complex)

    status = misc.StatusBar(sizes[comm.rank] * nk.prod(),
        title='sample coupling')

    scale = 2 * np.pi / nk

    for my_iq, iq in enumerate(range(*bounds[row.rank:row.rank + 2])):
        q1, q2, q3 = q[iq]
        Q1, Q2, Q3 = np.round(q[iq] / scale).astype(int)

        for K1 in range(nk[0]):
            KQ1 = (K1 + Q1) % nk[0]
            k1 = K1 * scale[0]

            for K2 in range(nk[1]):
                KQ2 = (K2 + Q2) % nk[1]
                k2 = K2 * scale[1]

                for K3 in range(nk[2]):
                    KQ3 = (K3 + Q3) % nk[2]
                    k3 = K3 * scale[2]

                    gqk = g(q1, q2, q3, k1, k2, k3, broadcast=False, comm=col)

                    if col.rank == 0:
                        if U is not None:
                            gqk = np.einsum('am,xab,bn->xmn',
                                U[KQ1, KQ2, KQ3].conj(), gqk, U[K1, K2, K3])

                        if u is not None:
                            gqk = np.einsum('xab,xu->uab', gqk, u[iq])

                        my_g[my_iq, :, K1, K2, K3, :, :] = gqk

                    status.update()

    node, images, g = MPI.shared_array((len(q), nph) + nk_orig + (nel, nel),
        dtype=complex, shared_memory=shared_memory, single_memory=not broadcast)

    if col.rank == 0:
        row.Gatherv(my_g, (g, row.gather(my_g.size)))

    col.Barrier() # should not be necessary

    if node.rank == 0:
        images.Bcast(g.view(dtype=float))

    node.Barrier()

    return g

def transform(g, q, nk, U=None, u=None, broadcast=True, shared_memory=False):
    """Transform q- and k-dependent coupling to band basis.

    See Also
    --------
    sample
    """
    sizes, bounds = MPI.distribute(len(q), bounds=True)

    nQ, nph, nk, nk, nel, nel = g.shape

    if U is not None:
        nel = U.shape[-1]

    if u is not None:
        nph = u.shape[-1]

    my_g = np.empty((sizes[comm.rank], nph, nk, nk, nel, nel), dtype=complex)

    scale = 2 * np.pi / nk

    for my_iq, iq in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        q1 = int(round(q[iq][0] / scale))
        q2 = int(round(q[iq][1] / scale))

        for k1 in range(nk):
            kq1 = (k1 + q1) % nk

            for k2 in range(nk):
                kq2 = (k2 + q2) % nk

                gqk = g[iq, :, k1, k2, :, :]

                if U is not None:
                    gqk = np.einsum('am,xab,bn->xmn',
                        U[kq1, kq2].conj(), gqk, U[k1, k2])

                if u is not None:
                    gqk = np.einsum('xab,xu->uab', gqk, u[iq])

                my_g[my_iq, :, k1, k2, :, :] = gqk

    node, images, g = MPI.shared_array((len(q), nph, nk, nk, nel, nel),
        dtype=complex, shared_memory=shared_memory, single_memory=not broadcast)

    comm.Gatherv(my_g, (g, comm.gather(my_g.size)))

    if node.rank == 0:
        images.Bcast(g.view(dtype=float))

    node.Barrier()

    return g

def q2r(elph, nq, nk, g, divide_ndegen=True):
    """Fourier-transform electron-phonon coupling from reciprocal to real space.

    Parameters
    ----------
    elph : object
        Localized model for electron-phonon coupling.
    nq, nk : tuple of int
        Number of q and k points along axes, i.e., shapes of uniform meshes.
    g : ndarray
        Electron-phonon coupling on complete uniform q- and k-point meshes.
    divide_ndegen : bool
        Divide real-space coupling by degeneracy of Wigner-Seitz point? Only
        ``True`` yields correct couplings. ``False`` should only be used for
        debugging.
    """
    nq_orig = nq
    nq = np.ones(3, dtype=int)
    nq[:len(nq_orig)] = nq_orig

    nk_orig = nk
    nk = np.ones(3, dtype=int)
    nk[:len(nk_orig)] = nk_orig

    g = np.reshape(g, (nq[0], nq[1], nq[2], elph.ph.size,
        nk[0], nk[1], nk[2], elph.el.size, elph.el.size))

    g = np.fft.ifftn(g.conj(), axes=(0, 1, 2, 4, 5, 6)).conj()

    if comm.rank == 0:
        for irg, (Rg1, Rg2, Rg3) in enumerate(elph.Rg):
            for irk, (Rk1, Rk2, Rk3) in enumerate(elph.Rk):
                elph.data[irg, :, irk, :, :] = g[
                    Rg1 % nq[0], Rg2 % nq[1], Rg3 % nq[2], :,
                    Rk1 % nk[0], Rk2 % nk[1], Rk3 % nk[2], :, :]

        if divide_ndegen:
            elph.divide_ndegen(elph.data)

    if elph.node.rank == 0:
        elph.images.Bcast(elph.data.view(dtype=float))

    comm.Barrier()

def coupling(filename, nQ, nmodes, nk, bands, Q=None, nq=None, offset=0,
        completion=True, complete_k=False, squeeze=False, status=False,
        phase=False):
    """Read and complete electron-phonon matrix elements."""

    if Q is not None:
        nQ = len(Q)
    else:
        Q = np.arange(nQ, dtype=int) + 1

    sizes = MPI.distribute(nQ)

    dtype = complex if phase else float

    elph = np.empty((nQ, nmodes, nk, nk, bands, bands), dtype=dtype)

    my_elph = np.empty((sizes[comm.rank], nmodes, nk, nk, bands, bands),
        dtype=dtype)

    my_elph[:] = np.nan

    my_Q = np.empty(sizes[comm.rank], dtype=int)
    comm.Scatterv((Q, sizes), my_Q)

    band_slice = slice(-5, -2) if phase else slice(-4, -1)

    for n, iq in enumerate(my_Q):
        if status:
            print('Read data for q point %d..' % iq)

        # TypeError: 'test' % 1
        # permitted: 'test' % np.array(1)

        with open(filename % iq) as data:
            for line in data:
                columns = line.split()

                if columns[0].startswith('#'):
                    continue

                k1, k2 = [int(i) - 1 for i in columns[:2]]
                jbnd, ibnd, nu = [int(i) - 1 for i in columns[band_slice]]

                ibnd -= offset
                jbnd -= offset

                if ibnd >= bands or jbnd >= bands:
                    continue

                indices = n, nu, k1, k2, ibnd, jbnd

                if phase:
                    my_elph[indices] \
                        = float(columns[-2]) + 1j * float(columns[-1])
                else:
                    my_elph[indices] = float(columns[-1])

    if completion:
        for n, iq in enumerate(my_Q):
            if status:
                print('Complete data for q point %d..' % iq)

            for nu in range(nmodes):
                for ibnd in range(bands):
                    for jbnd in range(bands):
                        bravais.complete(my_elph[n, nu, :, :, ibnd, jbnd])

    if complete_k and nq: # to be improved considerably
        comm.Gatherv(my_elph, (elph, sizes * nmodes * nk * nk * bands * bands))

        elph_complete = np.empty((nq, nq, nmodes, nk, nk, bands, bands),
            dtype=dtype)

        if comm.rank == 0:
            for nu in range(nmodes):
                for ibnd in range(bands):
                    for jbnd in range(bands):
                        elph_complete[:, :, nu, :, :, ibnd, jbnd] = (
                            bravais.complete_k(
                                elph[:, nu, :, :, ibnd, jbnd], nq))

        comm.Bcast(elph_complete)
        elph = elph_complete
    else:
        comm.Allgatherv(my_elph,
            (elph, sizes * nmodes * nk * nk * bands * bands))

    return elph[..., 0, 0] if bands == 1 and squeeze else elph

def read_EPW_output(epw_out, q, nq, nmodes, nk, bands=1, eps=1e-4,
        squeeze=False, status=False, epf=False, defpot=False):
    """Read electron-phonon coupling from EPW output file (using ``prtgkk``).

    Currently, the coupling must be defined on a uniform 2D k mesh
    (corresponding to triangular or square lattice).

    Parameters
    ----------
    epw_out : str
        Name of EPW output file.
    q : list of int
        List of q points as integer recriprocal lattice units.
    nq : int
        Number of q points per dimension.
    nmodes : int
        Number of phonon modes.
    nk : int
        Number of k points per dimension.
    bands : int
        Number of electronic bands.
    eps : float
        Tolerance for q and k points.
    squeeze : bool
        In single-band case, skip dimensions of output arrary corresponding to
        electronic bands?
    status : bool
        Report currently processed q point?
    epf : bool
        Read real and imaginary part of the coupling of dimension energy to the
        power of 2/3 from last two columns? A modified version of EPW is needed.
        Otherwise, the modulus of the coupling of dimension energy is read.
    defpot : bool
        Multiply coupling by square root of twice the phonon energy to obtain a
        quantity of dimension energy to the power of 2/3?
    """
    elph = np.empty((len(q), nmodes, nk, nk, bands, bands),
        dtype=complex if epf else float)

    q = [(q1, q2) for q1, q2 in q]

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

                    for _ in range(bands * bands * nmodes):
                        columns = next(data).split()

                        jbnd, ibnd, nu = [int(i) - 1 for i in columns[:3]]

                        if epf:
                            elph[iq, nu, k1, k2, ibnd, jbnd] = complex(
                                float(columns[-2]), float(columns[-1]))
                        elif defpot:
                            elph[iq, nu, k1, k2, ibnd, jbnd] = float(
                                columns[-1]) * np.sqrt(2 * float(columns[-2]))
                        else:
                            elph[iq, nu, k1, k2, ibnd, jbnd] = float(
                                columns[-1])

        if np.isnan(elph).any():
            print('Warning: EPW output incomplete!')

        if epf or defpot:
            elph *= 1e-3 ** 1.5 # meV^(3/2) to eV^(3/2)
        else:
            elph *= 1e-3 # meV to eV

    comm.Bcast(elph)

    return elph[..., 0, 0] if bands == 1 and squeeze else elph

def read_patterns(filename, q, nrep, status=True):
    """Read XML files with displacement patterns from QE."""

    if not hasattr(q, '__len__'):
        q = range(q)

    sizes = MPI.distribute(len(q))

    patterns = np.empty((len(q), nrep, nrep))

    my_patterns = np.empty((sizes[comm.rank], nrep, nrep))

    my_q = np.empty(sizes[comm.rank], dtype=int)
    comm.Scatterv((np.array(q), sizes), my_q)

    for my_iq, iq in enumerate(my_q):
        if status:
            print('Read displacement pattern for q point %d..' % (iq + 1))

        with open(filename % (iq + 1)) as data:
            def goto(pattern):
                for line in data:
                    if pattern in line:
                        return line

            goto('<NUMBER_IRR_REP ')
            if nrep != int(next(data)):
                print('Wrong number of representations!')

            for irep in range(nrep):
                goto('<DISPLACEMENT_PATTERN ')

                for jrep in range(nrep):
                    my_patterns[my_iq, irep, jrep] = float(
                        next(data).split(',')[0])

    comm.Allgatherv(my_patterns, (patterns, sizes * nrep * nrep))

    return patterns

def read_xml_files(filename, q, rep, bands, nbands, nk, squeeze=True,
        status=True, angle=120, angle0=0, old=False):
    """Read XML files with coupling in displacement basis from QE (*nosym*)."""

    if not hasattr(q, '__len__'):
        q = range(q)

    if not hasattr(rep, '__len__'):
        rep = range(rep)

    if not hasattr(bands, '__len__'):
        bands = [bands]

    a1, a2 = bravais.translations(angle, angle0)

    sizes = MPI.distribute(len(q))

    elph = np.empty((len(q), len(rep), nk, nk, len(bands), len(bands)),
        dtype=complex)

    my_elph = np.empty((sizes[comm.rank],
        len(rep), nk, nk, len(bands), len(bands)), dtype=complex)

    band_select = np.empty(nbands, dtype=int)
    band_select[:] = -1

    for n, m in enumerate(bands):
        band_select[m] = n

    my_q = np.empty(sizes[comm.rank], dtype=int)
    comm.Scatterv((np.array(q), sizes), my_q)

    for my_iq, iq in enumerate(my_q):
        if status:
            print('Read data for q point %d..' % (iq + 1))

        for my_irep, irep in enumerate(rep):
            with open(filename % (iq + 1, irep + 1)) as data:
                def goto(pattern):
                    for line in data:
                        if pattern in line:
                            return line

                def intag(tag):
                    return tag.split('>', 1)[1].split('<', 1)[0]

                tmp = goto('<NUMBER_OF_K')
                tmp = int(np.sqrt(int(next(data) if old else intag(tmp))))
                if nk != tmp:
                    print('Wrong number of k points!')

                tmp = goto('<NUMBER_OF_BANDS')
                tmp = int(next(data) if old else intag(tmp))
                if nbands != tmp:
                    print('Wrong number of bands!')

                for ik in range(nk * nk):
                    goto('<COORDINATES_XK')
                    k = list(map(float, next(data).split()))[:2]

                    k1 = int(round(np.dot(k, a1) * nk)) % nk
                    k2 = int(round(np.dot(k, a2) * nk)) % nk

                    goto('<PARTIAL_ELPH')

                    for n in band_select:
                        for m in band_select:
                            if n < 0 or m < 0:
                                next(data)
                            else:
                                my_elph[my_iq, my_irep, k1, k2, m, n] = complex(
                                    *list(map(float, next(data).split(','
                                        if old else None))))

    comm.Allgatherv(my_elph, (elph,
        sizes * len(rep) * nk * nk * len(bands) * len(bands)))

    return elph[..., 0, 0] if len(bands) == 1 and squeeze else elph

def write_xml_files(filename, data, angle=120, angle0=0):
    """Write XML files with coupling in displacement basis."""

    if comm.rank != 0:
        return

    if data.ndim == 4:
        data = data[:, :, np.newaxis, np.newaxis, :, :]

    nQ, nmodes, nk, nk, nbands, nbands = data.shape

    a1, a2 = bravais.translations(angle, angle0)
    b1, b2 = bravais.reciprocals(a1, a2)

    for iq in range(nQ):
        for irep in range(nmodes):
            with open(filename % (iq + 1, irep + 1), 'w') as xml:
                xml.write("""<?xml version="1.0"?>
<?iotk version="1.2.0"?>
<?iotk file_version="1.0"?>
<?iotk binary="F"?>
<?iotk qe_syntax="F"?>
<Root>
  <EL_PHON_HEADER>
    <DONE_ELPH type="logical" size="1">
      T
    </DONE_ELPH>
  </EL_PHON_HEADER>
  <PARTIAL_EL_PHON>
    <NUMBER_OF_K type="integer" size="1">
      %d
    </NUMBER_OF_K>
    <NUMBER_OF_BANDS type="integer" size="1">
      %d
    </NUMBER_OF_BANDS>""" % (nk * nk, nbands))

                ik = 0
                for k1 in range(nk):
                    for k2 in range(nk):
                        ik += 1

                        k = (k1 * b1 + k2 * b2) / nk

                        xml.write("""
    <K_POINT.%d>
      <COORDINATES_XK type="real" size="3" columns="3">
%23.15E %23.15E %23.15E
      </COORDINATES_XK>
      <PARTIAL_ELPH type="complex" size="%d">"""
                            % (ik, k[0], k[1], 0.0, nbands * nbands))

                        for j in range(nbands):
                            for i in range(nbands):
                                g = data[iq, irep, k1, k2, i, j]

                                xml.write("""
%23.15E,%23.15E""" % (g.real, g.imag))

                        xml.write("""
      </PARTIAL_ELPH>
    </K_POINT.%d>""" % ik)

                xml.write("""
  </PARTIAL_EL_PHON>
</Root>
""")

def write_data(filename, data):
    """Write array to ASCII file."""

    iterator = np.nditer(data, flags=['multi_index'])

    complex_data = np.iscomplexobj(data)

    integer_format = ' '.join('%%%dd' % len(str(n)) for n in data.shape)

    float_format = ' %16.9e'

    with open(filename, 'w') as text:
        text.write(integer_format % data.shape)

        text.write(' %s\n' % ('C' if complex_data else 'R'))

        for value in iterator:
            text.write(integer_format % iterator.multi_index)

            if complex_data:
                text.write(float_format % value.real)
                text.write(float_format % value.imag)
            else:
                text.write(float_format % value)

            text.write('\n')

def read_data(filename):
    """Read array to ASCII file."""

    with open(filename) as text:
        columns = text.next().split()
        shape = tuple(map(int, columns[:-1]))
        complex_data = {'R': False, 'C': True}[columns[-1]]

        data = np.empty(shape, dtype=complex if complex_data else float)

        ndim = len(shape)

        for line in text:
            columns = line.split()
            indices = tuple(map(int, columns[:ndim]))
            values = tuple(map(float, columns[ndim:]))

            if complex_data:
                data[indices] = complex(*values)
            else:
                data[indices] = values[0]

    return data
