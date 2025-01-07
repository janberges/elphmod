# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

# Some routines in this file follow wan2bloch.f90 of EPW v5.3.1.
# Copyright (C) 2010-2016 S. Ponce', R. Margine, C. Verdi, F. Giustino

"""Electron-phonon coupling from EPW."""

import numpy as np

import elphmod.bravais
import elphmod.dispersion
import elphmod.misc
import elphmod.MPI

comm = elphmod.MPI.comm
info = elphmod.MPI.info

class Model:
    r"""Localized model for electron-phonon coupling.

    Parameters
    ----------
    epmatwp : str
        File with electron-phonon coupling in localized bases from EPW.
    wigner : str
        File with definition of Wigner-Seitz supercells from EPW.
    el : :class:`elphmod.el.Model`
        Tight-binding model for the electrons.
    ph : :class:`elphmod.ph.Model`
        Mass-spring model for the phonons.
    Rk, Rg : ndarray
        Lattice vectors of Wigner-Seitz supercells if `wigner` is omitted.
    dk, dg : ndarray
        Degeneracies of Wigner-Seitz points if `wigner` is omitted.
    old_ws : bool
        Use previous definition of Wigner-Seitz cells? This is required if
        `patches/qe-6.3-backports.patch` has been used.
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
    el : :class:`elphmod.el.Model`
        Tight-binding model for the electrons.
    ph : :class:`elphmod.ph.Model`
        Mass-spring model for the phonons.
    Rk, Rg : ndarray
        Lattice vectors :math:`\vec R', \vec R` of Wigner-Seitz supercells.
    dk, dg : ndarray
        Degeneracies of Wigner-Seitz points.
    data : ndarray
        Corresponding electron-phonon matrix elements in Ry\ :sup:`3/2`.

        .. math::

            g_{\vec R i \vec R' \alpha \beta} = \frac \hbar {\sqrt M_i}
                \bra{0 \alpha}
                    \frac{\partial V}{\partial u_{\vec R i}}
                \ket{\vec R' \beta}

        If not :attr:`divide_mass`, the prefactor :math:`\hbar / \sqrt{M_i}` is
        absent and the units are Ry/bohr instead. If :attr:`ph.lr`, this is only
        the short-range component of the matrix elements.
    divide_mass : bool
        Has real-space coupling been divided by atomic masses?
    divide_ndegen : bool
        Has real-space coupling been divided by degeneracy of Wigner-Seitz
        point?
    node, images : MPI.Intracomm
        Communicators between processes that share memory or same ``node.rank``
        if `shared_memory`.
    q : ndarray
        Previously sampled q point, if any.
    gq : ndarray
        Rk-dependent coupling for above q point for possible reuse.
    cells : list of tuple of int, optional
        Lattice vectors of unit cells if the model describes a supercell.
    g0 : ndarray
        Coupling on original q and k meshes.
    Rk0 : int
        Index of electronic lattice vector at origin.
    """
    def g(self, q1=0, q2=0, q3=0, k1=0, k2=0, k3=0, elbnd=False, phbnd=False,
            broadcast=True, comm=comm):
        r"""Calculate electron-phonon coupling for arbitary points k and k + q.

        Parameters
        ----------
        q1, q2, q2 : float, default 0.0
            q point in crystal coordinates with period :math:`2 \pi`.
        k1, k2, k3 : float, default 0.0
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
        comm : MPI.Intracomm
            Group of processors running this function (for parallelization of
            Fourier transforms). Please note: To run this function serially,
            e.g., when parallelizing over q or k points, use ``elphmod.MPI.I``.

        Returns
        -------
        ndarray
            Fourier transform of :attr:`data`, possibly plus a long-range term
            and transformed into the band basis.
        """
        nRq, nph, nRk, nel, nel = self.data.shape

        q = np.array([q1, q2, q3])
        k = np.array([k1, k2, k3])

        if comm.allreduce(self.q is None or np.any(q != self.q)):
            self.q = q

            Rl, Ru = elphmod.MPI.distribute(nRq, bounds=True,
                comm=comm)[1][comm.rank:comm.rank + 2]

            my_g = np.einsum('Rxrab,R->xrab',
                self.data[Rl:Ru], np.exp(1j * self.Rg[Rl:Ru].dot(q)))

            # Sign convention in bloch2wan.f90 of EPW:
            # 1222  cfac = EXP(-ci * rdotk) / DBLE(nq)
            # 1223  epmatwp(:, :, :, :, ir) = epmatwp(:, :, :, :, ir)
            #           + cfac * epmatwe(:, :, :, :, iq)

            comm.Allreduce(my_g, self.gq)

            if self.ph.lr:
                g_lr = self.g_lr(q1, q2, q3)

                for a in range(self.el.size):
                    self.gq[:, self.Rk0, a, a] += g_lr

        Rl, Ru = elphmod.MPI.distribute(nRk, bounds=True,
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

        factor, vector = self.ph.generate_long_range(q1, q2, q3, perp=False)

        return np.einsum('g,gi->i', factor, vector)

    def gR(self, Rq1=0, Rq2=0, Rq3=0, Rk1=0, Rk2=0, Rk3=0):
        """Get electron-phonon matrix elements for arbitrary lattice vectors.

        Parameters
        ----------
        Rq1, Rq2, Rq3, Rk1, Rk2, Rk3 : int, default 0
            Lattice vectors in units of primitive vectors.

        Returns
        -------
        ndarray
            Element of :attr:`data` or zero.
        """
        index_q = elphmod.misc.vector_index(self.Rg, (Rq1, Rq2, Rq3))
        index_k = elphmod.misc.vector_index(self.Rk, (Rk1, Rk2, Rk3))

        if index_q is None or index_k is None:
            return np.zeros_like(self.data[0, :, 0, :, :])
        else:
            return self.data[index_q, :, index_k, :, :]

    def __init__(self, epmatwp=None, wigner=None, el=None, ph=None,
            Rk=None, dk=None, Rg=None, dg=None, old_ws=False,
            divide_mass=True, divide_ndegen=True, shared_memory=False):

        self.el = el
        self.ph = ph
        self.q = None
        self.g0 = None

        self.divide_mass = divide_mass
        self.divide_ndegen = divide_ndegen

        self.cells = [(0, 0, 0)]

        # read lattice vectors within Wigner-Seitz cell:

        if wigner is None:
            if Rk is None or dk is None or Rg is None or dg is None:
                return

            self.Rk, self.dk, self.Rg, self.dg = Rk, dk, Rg, dg
        else:
            self.Rk, self.dk, self.Rg, self.dg \
                = elphmod.bravais.read_wigner_file(wigner,
                    old_ws=old_ws, nat=ph.nat)

        # read coupling in Wannier basis from EPW output:
        # ('epmatwp' allocated and printed in 'ephwann_shuffle.f90')

        shape = len(self.Rg), ph.size, len(self.Rk), el.size, el.size

        self.node, self.images, self.data = elphmod.MPI.shared_array(shape,
            dtype=complex, shared_memory=shared_memory)

        if epmatwp is None:
            return

        if comm.rank == 0:
            with open(epmatwp, 'rb') as data:
                status = elphmod.misc.StatusBar((self.data.nbytes > 1e9)
                    * shape[0], title='load real-space coupling')

                for irg in range(shape[0]):
                    tmp = np.fromfile(data, dtype=np.complex128,
                        count=np.prod(shape[1:])).reshape(shape[1:])

                    self.data[irg] = np.swapaxes(tmp, 2, 3)

                    status.update()

                    # index orders:
                    # EPW (Fortran): a, b, R', x, R
                    # after read-in: R, x, R', b, a
                    # after transp.: R, x, R', a, b

                if np.fromfile(data).size:
                    print('Warning: File "%s" larger than expected!' % epmatwp)

            # undo supercell double counting:

            if divide_ndegen:
                self.divide_degeneracy(self.data)

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

        self.Rk0 = elphmod.misc.vector_index(self.Rk, (0, 0, 0))

    def divide_degeneracy(self, g):
        """Divide real-space coupling by degeneracy of Wigner-Seitz point.

        Parameters
        ----------
        g : ndarray
            Real-space coupling.
        """
        if self.dk is None or self.dg is None:
            info('Standardized coupling has no Wigner-Seitz cell!', error=True)

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

    def sample_orig(self, shared_memory=True):
        """Sample coupling on original q and k meshes."""

        if self.el.nk is None:
            info('Set "nk" attribute of electron model first!', error=True)

        if self.ph.q0 is None:
            self.ph.sample_orig()

        self.g0 = self.sample(self.ph.q0, self.el.nk,
            shared_memory=shared_memory)

    def update_short_range(self, shared_memory=True):
        """Update short-range part of real-space coupling."""

        if self.g0 is None:
            info('Run "sample_orig" before changing Z, Q, etc.!', error=True)

        if not self.ph.lr:
            q2r(self, self.ph.nq, self.el.nk, self.g0, None, self.divide_mass)
            return

        self.ph.prepare_long_range()

        g = elphmod.MPI.SharedArray(self.g0.shape, dtype=complex,
            shared_memory=shared_memory)

        g_lr = elphmod.dispersion.sample(self.g_lr, self.ph.q0)

        for _ in range(len(self.el.nk)):
            g_lr = g_lr[..., np.newaxis]

        if comm.rank == 0:
            g[...] = self.g0[...]

            for a in range(self.el.size):
                g[..., a, a] -= g_lr

        g.Bcast()

        q2r(self, self.ph.nq, self.el.nk, g, None, self.divide_mass)

    def sample(self, *args, **kwargs):
        """Sample coupling.

        See also
        --------
        sample
        """
        return sample(self.g, *args, **kwargs)

    def supercell(self, N1=1, N2=1, N3=1, shared_memory=False, sparse=False):
        """Map localized model for electron-phonon coupling onto supercell.

        Parameters
        ----------
        N1, N2, N3 : tuple of int or int, default 1
            Supercell lattice vectors in units of primitive lattice vectors.
        shared_memory : bool, default False
            Store mapped coupling in shared memory?
        sparse : bool, default False
            Only calculate q = k = 0 coupling as a list of sparse matrices to
            save memory? The result, which is assumed to be real, is stored in
            the attribute :attr:`gs`. Consider using :meth:`standardize` with
            nonzero `eps` and `symmetrize` before.

        Returns
        -------
        object
            Localized model for electron-phonon coupling for supercell.

        See Also
        --------
        elphmod.bravais.supercell
        """
        elph = Model(
            el=self.el.supercell(N1, N2, N3, sparse=sparse),
            ph=self.ph.supercell(N1, N2, N3, sparse=sparse))

        supercell = elphmod.bravais.supercell(N1, N2, N3)
        elph.cells = supercell[-1]

        elph.divide_mass = self.divide_mass
        elph.divide_ndegen = self.divide_ndegen

        const = dict()

        if sparse:
            sparse_array = elphmod.misc.get_sparse_array()

            elph.gs = [sparse_array((elph.el.size, elph.el.size))
                for x in range(elph.ph.size)]

            if elph.ph.lr:
                g_lr = elph.g_lr().real

                for x in range(elph.ph.size):
                    elph.gs[x].setdiag(g_lr[x])

            if abs(self.data.imag).sum() > 1e-6 * abs(self.data.real).sum():
                info('Warning: Significant imaginary part of coupling ignored!')

        Rg = np.empty((len(self.Rg), 3), dtype=int)
        Rk = np.empty((len(self.Rk), 3), dtype=int)

        rg = np.empty(len(self.Rg), dtype=int)
        rk = np.empty(len(self.Rk), dtype=int)

        status = elphmod.misc.StatusBar(-(-len(elph.cells) // comm.size)
            * len(self.Rg), title='map coupling onto supercell')

        for i in range(len(elph.cells)):
            if i % comm.size != comm.rank:
                continue

            for g in range(len(self.Rg)):
                Rg[g], rg[g] = elphmod.bravais.to_supercell(self.Rg[g]
                    + elph.cells[i], supercell)

            for k in range(len(self.Rk)):
                Rk[k], rk[k] = elphmod.bravais.to_supercell(self.Rk[k]
                    + elph.cells[i], supercell)

            A = i * self.el.size

            for g in range(len(self.Rg)):
                X = rg[g] * self.ph.size

                for k in range(len(self.Rk)):
                    B = rk[k] * self.el.size

                    if sparse:
                        for x in range(self.ph.size):
                            elph.gs[X + x][
                                A:A + self.el.size,
                                B:B + self.el.size] += self.data[g, x, k].real
                        continue

                    R = (*Rg[g], *Rk[k])

                    if R not in const:
                        const[R] = np.zeros((elph.ph.size,
                            elph.el.size, elph.el.size), dtype=complex)

                    const[R][
                        X:X + self.ph.size,
                        A:A + self.el.size,
                        B:B + self.el.size] = self.data[g, :, k]

                status.update()

        if sparse:
            # DOK/CSR format efficient for matrix construction/calculations:

            for x in range(elph.ph.size):
                elph.gs[x] = comm.allreduce(elph.gs[x]).tocsr()

            elph.gs = np.array(elph.gs)

            if elphmod.misc.verbosity >= 2 and comm.rank == 0:
                import pickle

                print('Sparse representation of coupling requires %.6f GB'
                    % (len(pickle.dumps(elph.gs)) / 1e9))

            return elph

        elph.Rg = np.array(sorted(set().union(*comm.allgather([R[:3]
            for R in const])))).reshape((-1, 3))

        elph.Rk = np.array(sorted(set().union(*comm.allgather([R[3:]
            for R in const])))).reshape((-1, 3))

        elph.node, elph.images, elph.data = elphmod.MPI.shared_array(
            (len(elph.Rg), elph.ph.size, len(elph.Rk), elph.el.size,
            elph.el.size), dtype=complex, shared_memory=shared_memory)

        elph.gq = np.empty((elph.ph.size, len(elph.Rk),
            elph.el.size, elph.el.size), dtype=complex)

        elph.Rk0 = elphmod.misc.vector_index(elph.Rk, (0, 0, 0))

        if elph.node.rank == 0:
            elph.data[...] = 0.0

        status = elphmod.misc.StatusBar(len(elph.Rg) * len(elph.Rk),
            title='convert supercell coupling to standard format')

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
            # Reduce chunk-wise to avoid integer overflow for message size:

            for g in range(len(elph.Rg)):
                elph.images.Allreduce(elphmod.MPI.MPI.IN_PLACE,
                    elph.data[g].view(dtype=float))

        return elph

    def standardize(self, eps=0.0, symmetrize=False):
        r"""Standardize real-space coupling data.

        - Keep only nonzero coupling matrices.
        - Sum over repeated lattice vectors.
        - Sort lattice vectors.
        - Optionally symmetrize coupling:

        .. math::

            g_{\vec q, \vec k} = g_{-\vec q, \vec k + \vec q}^\dagger,
            g_{\vec R, \vec R'} = g_{\vec R - \vec R', -\vec R'}^\dagger

        Parameters
        ----------
        eps : float
            Threshold for "nonzero" matrix elements in units of the maximum
            matrix element.
        symmetrize : bool
            Symmetrize coupling?
        """
        if comm.rank == 0:
            if eps:
                self.data[abs(self.data) < eps * abs(self.data).max()] = 0.0

            const = dict()

            const[0, 0, 0, 0, 0, 0] = np.zeros((self.ph.size,
                self.el.size, self.el.size), dtype=complex)

            status = elphmod.misc.StatusBar(len(self.Rg) * len(self.Rk),
                title='standardize coupling data')

            for g in range(len(self.Rg)):
                for k in range(len(self.Rk)):
                    if np.any(self.data[g, :, k] != 0):
                        R = (*self.Rg[g], *self.Rk[k])

                        if R in const:
                            const[R] += self.data[g, :, k]
                        else:
                            const[R] = self.data[g, :, k].copy()

                        if symmetrize:
                            R = (*self.Rg[g] - self.Rk[k], *-self.Rk[k])

                            g_adj = self.data[g, :, k].swapaxes(1, 2).conj()

                            if R in const:
                                const[R] += g_adj
                            else:
                                const[R] = g_adj

                    status.update()

            Rg = sorted(set(R[:3] for R in const))
            Rk = sorted(set(R[3:] for R in const))

            ng = len(Rg)
            nk = len(Rk)
        else:
            ng = nk = None

        ng = comm.bcast(ng)
        nk = comm.bcast(nk)

        if ng <= len(self.Rg):
            self.Rg = self.Rg[:ng]
        else:
            self.Rg = np.empty((ng, 3), dtype=int)

        if nk <= len(self.Rk):
            self.Rk = self.Rk[:nk]
        else:
            self.Rk = np.empty((ng, 3), dtype=int)

        shape = ng, self.ph.size, nk, self.el.size, self.el.size

        if np.prod(shape) <= self.data.size:
            self.data = self.data.ravel()
            self.data = self.data[:np.prod(shape)]
            self.data = self.data.reshape(shape)
        else:
            self.node, self.images, self.data = elphmod.MPI.shared_array(shape,
                dtype=complex, shared_memory=self.node.size > 1)

        if comm.rank == 0:
            self.Rg[...] = Rg
            self.Rk[...] = Rk

            self.data[...] = 0.0

            for g in range(len(Rg)):
                for k in range(len(Rk)):
                    if Rg[g] + Rk[k] in const:
                        self.data[g, :, k] = const[Rg[g] + Rk[k]]

                        if symmetrize:
                            self.data[g, :, k] /= 2

        comm.Bcast(self.Rg)
        comm.Bcast(self.Rk)

        if self.node.rank == 0:
            self.images.Bcast(self.data.view(dtype=float))

        self.dk = None
        self.dg = None
        self.q = None
        self.gq = np.empty((self.ph.size, len(self.Rk),
            self.el.size, self.el.size), dtype=complex)

        self.Rk0 = elphmod.misc.vector_index(self.Rk, (0, 0, 0))

        comm.Barrier()

    def symmetrize(self):
        """Symmetrize electron-phonon coupling."""

        self.standardize(symmetrize=True)

    def asr(self, report=True):
        """Apply acoustic sum rule correction to electron-phonon coupling.

        This will suppress all coupling of electrons to acoustic phonon modes.
        The matrix elements are subject to a constant relative change so that
        zeros remain zeros and the largest values change the most. There might
        be a better way to accomplish this.

        report : bool
            Print sums before and after correction?
        """
        if comm.rank == 0:
            shape = self.data.shape

            self.data = self.data.reshape((len(self.Rg) * self.ph.nat, -1))

            zero = self.data.sum(axis=0)

            if report:
                print('Acoustic sum (before): %g' % abs(zero).max())

            norm = abs(self.data).sum(axis=0)

            norm[norm == 0] = 1.0 # corresponding "zero" really 0 by definition

            self.data -= abs(self.data) / norm * zero

            if report:
                zero = self.data.sum(axis=0)

                print('Acoustic sum (after): %g' % abs(zero).max())

            self.data = self.data.reshape(shape)

        if self.node.rank == 0:
            self.images.Bcast(self.data.view(dtype=float))

        comm.Barrier()

    def decay_epmate(self):
        """Plot maximum matrix element as a function of hopping distance.

        Use ``divide_mass=False`` to recreate EPW's *decay.epmate* file.

        Returns
        -------
        ndarray
            Distances.
        ndarray
            Maximum absolute matrix elements.
        """
        d = np.linalg.norm(self.Rk.dot(self.ph.a), axis=1) * elphmod.misc.a0
        g = abs(self.data).max(axis=(0, 1, 3, 4))

        nonzero = np.where(g > 0)

        return d[nonzero], g[nonzero]

    def decay_epmatp(self):
        """Plot maximum matrix element as a function of displacement distance.

        Use ``divide_mass=False`` to recreate EPW's *decay.epmatp* file.

        Returns
        -------
        ndarray
            Distances.
        ndarray
            Maximum absolute matrix elements.
        """
        d = np.linalg.norm(self.Rg.dot(self.ph.a), axis=1) * elphmod.misc.a0
        g = abs(self.data).max(axis=(1, 2, 3, 4))

        nonzero = np.where(g > 0)

        return d[nonzero], g[nonzero]

    def clear(self):
        """Delete all lattice vectors and associated matrix elements."""

        self.el.clear()
        self.ph.clear()

        self.Rk = np.zeros_like(self.Rk[:0, :])
        self.Rg = np.zeros_like(self.Rg[:0, :])

        self.data = np.zeros_like(self.data[:0, :, :0, :, :])
        self.gq = np.zeros_like(self.gq[:, :0, :, :])

    def to_epmatwp(self, prefix):
        """Save coupling to *.epmatwp* and *.wigner* files.

        If :attr:`divide_ndegen`, the division by the degeneracies of the
        Wigner-Seitz points is not undone before writing the *.epmatwp* file.
        Instead, all degeneracies in the *.wiger* file are set to one.

        Parameters
        ----------
        prefix : str
            Filename stem.
        """
        if comm.rank == 0:
            with open('%s.wigner' % prefix, 'w') as data:
                if self.divide_ndegen:
                    dims = dims2 = 1
                    dk = np.ones((dims, dims, len(self.Rk)), dtype=int)
                    dg = np.ones((1, dims, dims2, len(self.Rg)), dtype=int)
                else:
                    dims, dims2 = self.dg.shape[1:3]
                    dk = self.dk
                    dg = self.dg

                data.write('%d 0 %d %d %d\n'
                    % (len(self.Rk), len(self.Rg), dims, dims2))

                for ir in range(len(self.Rk)):
                    data.write('%6d %5d %5d\n' % tuple(self.Rk[ir]))

                    for iw in range(dims):
                        data.write(' '.join('%d' % dk[iw2, iw, ir]
                            for iw2 in range(dims)) + '\n')

                for ir in range(len(self.Rg)):
                    data.write('%6d %5d %5d\n' % tuple(self.Rg[ir]))

                    for iw in range(dims):
                        data.write(' '.join('%d' % dg[0, iw, na, ir]
                            for na in range(dims2)) + '\n')

            with open('%s.epmatwp' % prefix, 'wb') as data:
                epmatwp = self.data.reshape((len(self.Rg), self.ph.nat, 3,
                    len(self.Rk), self.el.size, self.el.size))

                for g in range(len(self.Rg)):
                    for na in range(self.ph.nat):
                        if self.divide_mass:
                            buf = epmatwp[g, na] * np.sqrt(self.ph.M[na])
                        else:
                            buf = epmatwp[g, na]

                        buf.swapaxes(-2, -1).astype(np.complex128).tofile(data)

def sample(g, q, nk=None, U=None, u=None, squared=False, broadcast=True,
        shared_memory=False):
    r"""Sample coupling for given q and k points and transform to band basis.

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
    squared : bool
        Sample squared complex modulus instead? This is more memory-efficient
        than sampling the complex coupling and taking the squared modulus later.
    broadcast : bool
        Broadcast result from rank 0 to all processes?
    shared_memory : bool, optional
        Store transformed coupling in shared memory?
    """
    sizes, bounds = elphmod.MPI.distribute(len(q), bounds=True)
    col, row = elphmod.MPI.matrix(len(q))

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
        U = np.reshape(U, (*nk, -1, nel))

    if u is not None:
        nph = u.shape[-1]

    my_g = np.empty((sizes[row.rank], nph, *nk, nel, nel),
        dtype=float if squared else complex)

    status = elphmod.misc.StatusBar(sizes[comm.rank] * nk.prod(),
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

                        if squared:
                            gqk *= gqk.conj()
                            gqk = gqk.real

                        my_g[my_iq, :, K1, K2, K3, :, :] = gqk

                    status.update()

    node, images, g = elphmod.MPI.shared_array((len(q), nph, *nk_orig,
        nel, nel), dtype=my_g.dtype, shared_memory=shared_memory,
        single_memory=not broadcast)

    if node.size == comm.size > 1 and broadcast: # shared memory on single node

        # As Gatherv into shared memory can require more memory than expected
        # and lead to segmentation faults, we use a different approach here.

        if col.rank == 0:
            for column in range(row.size):
                if column == row.rank:
                    g[bounds[row.rank]:bounds[row.rank + 1]] = np.reshape(my_g,
                        g[bounds[row.rank]:bounds[row.rank + 1]].shape)
                    del my_g

                row.Barrier()

        return g

    if col.rank == 0:
        row.Gatherv(my_g, (g, row.gather(my_g.size)))

    col.Barrier() # should not be necessary

    if node.rank == 0:
        for iq in range(len(q)):
            images.Bcast(g[iq].view(dtype=float))

    node.Barrier()

    return g

def transform(g, q, nk, U=None, u=None, squared=False, broadcast=True,
        shared_memory=False):
    """Transform q- and k-dependent coupling to band basis.

    See Also
    --------
    sample
    """
    sizes, bounds = elphmod.MPI.distribute(len(q), bounds=True)

    nQ, nph, nk, nk, nel, nel = g.shape

    if U is not None:
        nel = U.shape[-1]

    if u is not None:
        nph = u.shape[-1]

    my_g = np.empty((sizes[comm.rank], nph, nk, nk, nel, nel),
        dtype=float if squared else complex)

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

                if squared:
                    gqk *= gqk.conj()
                    gqk = gqk.real

                my_g[my_iq, :, k1, k2, :, :] = gqk

    node, images, g = elphmod.MPI.shared_array((len(q), nph, nk, nk, nel, nel),
        dtype=my_g.dtype, shared_memory=shared_memory,
        single_memory=not broadcast)

    comm.Gatherv(my_g, (g, comm.gather(my_g.size)))

    if node.rank == 0:
        images.Bcast(g.view(dtype=float))

    node.Barrier()

    return g

def q2r(elph, nq, nk, g, r=None, divide_mass=True, shared_memory=False):
    """Fourier-transform electron-phonon coupling from reciprocal to real space.

    Parameters
    ----------
    elph : :class:`Model`
        Localized model for electron-phonon coupling.
    nq, nk : tuple of int
        Number of q and k points along axes, i.e., shapes of uniform meshes.
    g : ndarray
        Electron-phonon coupling on complete uniform q- and k-point meshes.
    r : ndarray, optional
        Positions of orbital centers. If given, the Wigner-Seitz lattice vectors
        are determined again, whereby the distances to the displaced atom and
        the initial orbital are are both measured from the final orbital in the
        unit cell at the origin (first orbital index). This argument is required
        when changing `nq` or `nk`.
    divide_mass : bool, default True
        Has input coupling been divided by square root of atomic mass? This is
        independent of ``elph.divide_mass``, which is always respected.
    shared_memory : bool, default False
        Store real-space coupling in shared memory?
    """
    nq_orig = nq
    nq = np.ones(3, dtype=int)
    nq[:len(nq_orig)] = nq_orig

    nk_orig = nk
    nk = np.ones(3, dtype=int)
    nk[:len(nk_orig)] = nk_orig

    g = np.reshape(g, (*nq, elph.ph.size, *nk, elph.el.size, elph.el.size))

    M = np.mgrid[:elph.ph.size, :elph.el.size, :elph.el.size].reshape((3, -1)).T

    sizes, bounds = elphmod.MPI.distribute(len(M), bounds=True)

    my_data = np.empty((sizes[comm.rank], *nq, *nk), dtype=complex)

    for i, (nu, m, n) in enumerate(M[bounds[comm.rank]:bounds[comm.rank + 1]]):
        my_data[i] = np.fft.ifftn(g[:, :, :, nu, :, :, :, m, n].conj()).conj()

    if comm.rank == 0:
        data = np.empty((elph.ph.size, elph.el.size, elph.el.size, *nq, *nk),
            dtype=complex)
    else:
        data = None

    comm.Gatherv(my_data, (data, comm.gather(my_data.size)))

    if r is not None:
        elph.Rg, ndegen_g, wslen = elphmod.bravais.wigner(*nq,
            elph.ph.a, r, elph.ph.r)

        elph.dg = ndegen_g.transpose(1, 2, 0)[np.newaxis]

        elph.Rk, ndegen_k, wslen = elphmod.bravais.wigner(*nk, elph.ph.a, r)

        elph.dk = ndegen_k.transpose()

        elph.node, elph.images, elph.data = elphmod.MPI.shared_array(
            (len(elph.Rg), elph.ph.size, len(elph.Rk), elph.el.size,
            elph.el.size), dtype=complex, shared_memory=shared_memory)

    if comm.rank == 0:
        for irg, (g1, g2, g3) in enumerate(elph.Rg % nq):
            for irk, (k1, k2, k3) in enumerate(elph.Rk % nk):
                elph.data[irg, :, irk] = data[..., g1, g2, g3, k1, k2, k3]

        if elph.divide_ndegen:
            elph.divide_degeneracy(elph.data)

        if divide_mass and not elph.divide_mass:
            for x in range(elph.ph.size):
                elph.data[:, x] *= np.sqrt(elph.ph.M[x // 3])

        elif not divide_mass and elph.divide_mass:
            for x in range(elph.ph.size):
                elph.data[:, x] /= np.sqrt(elph.ph.M[x // 3])

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

    sizes = elphmod.MPI.distribute(nQ)

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
            print('Read data for q point %d' % iq)

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
                print('Complete data for q point %d' % iq)

            for nu in range(nmodes):
                for ibnd in range(bands):
                    for jbnd in range(bands):
                        elphmod.bravais.complete(my_elph[n, nu, :, :,
                            ibnd, jbnd])

    if complete_k and nq: # to be improved considerably
        comm.Gatherv(my_elph, (elph, sizes * nmodes * nk * nk * bands * bands))

        elph_complete = np.empty((nq, nq, nmodes, nk, nk, bands, bands),
            dtype=dtype)

        if comm.rank == 0:
            for nu in range(nmodes):
                for ibnd in range(bands):
                    for jbnd in range(bands):
                        elph_complete[:, :, nu, :, :, ibnd, jbnd] = (
                            elphmod.bravais.complete_k(
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

                    for jbnd in range(bands):
                        for ibnd in range(bands):
                            for nu in range(nmodes):
                                columns = next(data).split()

                                if epf:
                                    elph[iq, nu, k1, k2, ibnd, jbnd] = complex(
                                        float(columns[-2]), float(columns[-1]))
                                elif defpot:
                                    elph[iq, nu, k1, k2, ibnd, jbnd] = float(
                                        columns[-1]) * np.sqrt(
                                            2 * float(columns[-2]))
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

def read_prtgkk(epw_out, nq, nmodes, nk, nbnd):
    """Read frequencies and coupling from EPW output (using ``prtgkk``).

    Parameters
    ----------
    epw_out : str
        Name of EPW output file.
    nq : int
        Number of q points.
    nmodes : int
        Number of phonon modes.
    nk : int
        Number of k points.
    nbnd : int
        Number of electronic bands.

    Returns
    -------
    ndarray
        Phonon frequencies (meV).
    ndarray
        Electron-phonon coupling (meV).
    """
    w = np.empty((nq, nmodes))
    g = np.empty((nq, nmodes, nk, nbnd, nbnd))

    if comm.rank == 0:
        iq = -1

        with open(epw_out) as lines:
            for line in lines:
                if line.startswith('     iq = '):
                    ik = -1
                    iq += 1

                if line.startswith('     ik = '):
                    ik += 1

                    next(lines)
                    next(lines)

                    for jbnd in range(nbnd):
                        for ibnd in range(nbnd):
                            for nu in range(nmodes):
                                columns = next(lines).split()

                                w[iq, nu] = float(columns[-2])
                                g[iq, nu, ik, ibnd, jbnd] = float(columns[-1])

    comm.Bcast(w)
    comm.Bcast(g)

    return w, g

def read_L(epw_out):
    """Read range-separation parameter from EPW output

    Parameters
    ----------
    epw_out : str
        Name of EPW output file.

    Returns
    -------
    float
        Range-separation parameter if found, ``None`` otherwise.
    """
    L = None

    if comm.rank == 0:
        with open(epw_out) as lines:
            for line in lines:
                columns = line.split()

                if columns and columns[0] == 'L':
                    L = float(columns[1])

    return comm.bcast(L)

def read_patterns(filename, q, nrep, status=True):
    """Read XML files with displacement patterns from QE."""

    if not hasattr(q, '__len__'):
        q = range(q)

    sizes = elphmod.MPI.distribute(len(q))

    patterns = np.empty((len(q), nrep, nrep))

    my_patterns = np.empty((sizes[comm.rank], nrep, nrep))

    my_q = np.empty(sizes[comm.rank], dtype=int)
    comm.Scatterv((np.array(q), sizes), my_q)

    for my_iq, iq in enumerate(my_q):
        if status:
            print('Read displacement pattern for q point %d' % (iq + 1))

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

    a1, a2 = elphmod.bravais.translations(angle, angle0)

    sizes = elphmod.MPI.distribute(len(q))

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
            print('Read data for q point %d' % (iq + 1))

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

    a1, a2 = elphmod.bravais.translations(angle, angle0)
    b1, b2 = elphmod.bravais.reciprocals(a1, a2)

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
    """Read array from ASCII file."""

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

def ph2epw(fildyn='dyn', outdir='work', dvscf_dir='save'):
    """Convert PHonon output to EPW input.

    Based on script `pp.py` provided with EPW code (C) 2015 Samuel Ponce.

    All arguments can be overwritten by environment variables of the same name.

    Parameters
    ----------
    dyn : str
        Prefix of dynamical-matrix files.
    outdir : str
        QE output directory.
    dvscf_dir : str
        EPW input directory.
    """
    import glob
    import os
    import shutil

    if 'fildyn' in os.environ:
        fildyn = os.environ['fildyn']

    if 'outdir' in os.environ:
        outdir = os.environ['outdir']

    if 'dvscf_dir' in os.environ:
        dvscf_dir = os.environ['dvscf_dir']

    if fildyn.endswith('.xml'):
        fildyn = fildyn[:-4]

    ext = '.xml' if os.path.isfile('%s1.xml' % fildyn) else ''

    if not os.path.isfile('%s1%s' % (fildyn, ext)) or not os.path.isdir(outdir):
        info('Usage: [fildyn=...] [outdir=...] [dvscf_dir=...] ph2epw',
            error=True)

    phsave, = glob.glob('%s/_ph0/*.phsave' % outdir)
    prefix = phsave[len('%s/_ph0/' % outdir):-len('.phsave')]

    try:
        shutil.copytree(phsave, '%s/%s.phsave' % (dvscf_dir, prefix))
    except FileExistsError:
        info('Warning: EPW dvscf_dir "%s" already exists!' % dvscf_dir)
        return

    n = 0
    while True:
        n += 1

        dyn = '%s%d%s' % (fildyn, n, ext)

        if not os.path.isfile(dyn):
            break

        shutil.copy2(dyn, '%s/%s.dyn_q%d%s' % (dvscf_dir, prefix, n, ext))

        orig = '%s/_ph0' % outdir

        if n > 1:
            orig += '/%s.q_%d' % (prefix, n)

        for suffix in 'dvscf', 'dvscf_paw':
            origfile = '%s/%s.%s1' % (orig, prefix, suffix)

            if os.path.isfile(origfile):
                shutil.copy2(origfile, '%s/%s.%s_q%d'
                    % (dvscf_dir, prefix, suffix, n))
