#!/usr/bin/env python3

# Copyright (C) 2017-2022 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import numpy as np

from . import bravais, dispersion, misc, MPI, occupations
comm = MPI.comm
info = MPI.info

class Model(object):
    """Tight-binding model for the electrons.

    Parameters
    ----------
    seedname : str
        Common prefix of Wannier90 output files: *seedname_hr.dat* with the
        Hamiltonian in the Wannier basis and, optionally, *seedname_wsvec.dat*
        with superlattice vectors used to symmetrize the long-range hopping.
    divide_ndegen : bool, default True
        Divide hopping by degeneracy of Wigner-Seitz point and apply the
        abovementioned correction? Only ``True`` yields correct bands.
        ``False`` is only used in combination with :func:`decayH`.
    read_xsf : bool, default False
        Read Wannier functions in position representation in the XCrySDen
        structure format (XSF)?
    normalize_wf : bool, default False
        Normalize Wannier functions? This is only recommended if a single and
        complete image of each Wannier function is contained in the supercell
        written to the XSF file. This may not be the case for a small number of
        k points along one direction (e.g., a single k point in the out-of-plane
        direction in the case of 2D materials), especially in combination with a
        large supercell.
    buffer_wf : bool, default False
        After reading XSF files, store Wannier functions and corresponding
        real-space mesh in binary files, which will be read next time? Since
        reading the XSF files is slow, this can save a lot of time for large
        systems. Make sure to delete the binary files *seedname_wf.npy* and
        *seedname_xyz.npy* whensoever the XSF files change.
    check_ortho : bool, default False
        Check if Wannier functions are orthogonal?
    shared_memory : bool, default False
        Store Wannier functions in shared memory?

    Attributes
    ----------
    R : ndarray
        Lattice vectors of Wigner-Seitz supercell.
    data : ndarray
        Corresponding on-site energies and hoppings.
    size : int
        Number of Wannier functions/bands.
    cells : list of tuple of int, optional
        Lattice vectors of unit cells if the model describes a supercell.
    N : list of tuple of int, optional
        Primitive vectors of supercell if the model describes a supercell.
    W : ndarray, optional
        Wannier functions in position representation if `read_xsf`.
    r : ndarray, optional
        Cartesian positions belonging to Wannier functions if `read_xsf`.
    tau : ndarray, optional
        Positions of basis atoms if `read_xsf`.
    atom_order : list of str, optional
        Ordered list of atoms if `read_xsf`.
    dV : float, optional
        Volume element/voxel volume belonging to `r` if `read_xsf`.
    """
    def H(self, k1=0, k2=0, k3=0):
        """Set up Hamilton operator for arbitrary k point."""

        k = np.array([k1, k2, k3])

        # Sign convention in hamiltonian.f90 of Wannier90:
        # 295  fac=exp(-cmplx_i*rdotk)/real(num_kpts,dp)
        # 296  ham_r(:,:,irpt)=ham_r(:,:,irpt)+fac*ham_k(:,:,loop_kpt)

        # Note that the data from Wannier90 can be interpreted like this:
        # self.data[self.R == R - R', a, b] = <R' a|H|R b> = <R b|H|R' a>

        # Compare this convention [doi:10.26092/elib/250, Eq. 2.35a]:
        # t(R - R', a, b) = <R a|H|R' b> = <R' b|H|R a>

        return np.einsum('Rab,R->ab', self.data, np.exp(1j * self.R.dot(k)))

    def t(self, R1=0, R2=0, R3=0):
        """Get on-site or hopping energy for arbitrary lattice vector."""

        index = misc.vector_index(self.R, (R1, R2, R3))

        if index is None:
            return np.zeros_like(self.data[0])
        else:
            return self.data[index]

    def __init__(self, seedname=None, divide_ndegen=True, read_xsf=False,
            normalize_wf=False, buffer_wf=False, check_ortho=False,
            shared_memory=False):

        if seedname is None:
            return

        if seedname.endswith('_hr.dat'):
            seedname = seedname[:-7]

        self.R, self.data = read_hrdat('%s_hr.dat' % seedname, divide_ndegen)
        self.size = self.data.shape[1]

        supvecs = read_wsvecdat('%s_wsvec.dat' % seedname)

        if supvecs is not None and divide_ndegen:
            if comm.rank == 0:
                const = dict()

                for n, (i, j, k) in enumerate(self.R):
                    for a in range(self.size):
                        for b in range(self.size):
                            key = i, j, k, a, b
                            t = self.data[n, a, b] / len(supvecs[key])

                            for I, J, K in supvecs[key]:
                                R = i + I, j + J, k + K

                                if R not in const:
                                    const[R] = np.zeros((self.size, self.size),
                                        dtype=complex)

                                const[R][a, b] += t

                count = len(const)

                self.R = np.array(list(const.keys()), dtype=int)
                self.data = np.array(list(const.values()))
            else:
                count = None

            count = comm.bcast(count)

            if comm.rank != 0:
                self.R = np.empty((count, 3), dtype=int)
                self.data = np.empty((count, self.size, self.size),
                    dtype=complex)

            comm.Bcast(self.R)
            comm.Bcast(self.data)

        if read_xsf:
            if buffer_wf:
                self.W = MPI.load('%s_wf.npy' % seedname, shared_memory)
                self.r = MPI.load('%s_xyz.npy' % seedname, shared_memory)

            read_buffer = buffer_wf and self.W.size and self.r.size

            r0, a, self.atom_order, self.tau, shape = misc.read_xsf(
                '%s_head.xsf' % seedname if read_buffer else
                '%s_%05d.xsf' % (seedname, 1), only_header=True)

            self.dV = abs(np.dot(np.cross(a[0], a[1]), a[2])) / np.prod(shape)

            if not read_buffer:
                self.r = misc.real_space_grid(shape, r0, a, shared_memory)

                sizes, bounds = MPI.distribute(self.size, bounds=True)

                my_W = np.empty((sizes[comm.rank],) + shape)
                node, images, self.W = MPI.shared_array((self.size,) + shape,
                    shared_memory=shared_memory)

                for my_n, n in enumerate(range(
                        *bounds[comm.rank:comm.rank + 2])):

                    my_W[my_n] = misc.read_xsf('%s_%05d.xsf'
                        % (seedname, n + 1), comm=MPI.I)[-1]

                    if normalize_wf:
                        my_W[my_n] /= np.sqrt(np.sum(my_W[my_n] ** 2) * self.dV)

                comm.Gatherv(my_W, (self.W, sizes * np.prod(shape)))

                if node.rank == 0:
                    images.Bcast(self.W)

                comm.Barrier()

                if buffer_wf and comm.rank == 0:
                    misc.write_xsf('%s_head.xsf' % seedname, r0, a,
                        self.atom_order, self.tau, shape, only_header=True)

                    np.save('%s_wf.npy' % seedname, self.W)
                    np.save('%s_xyz.npy' % seedname, self.r)

            if check_ortho:
                info('Check if Wannier functions are orthogonal:')
                for m in range(self.size):
                    for n in range(self.size):
                        if (m * self.size + n) % comm.size == comm.rank:
                            print('%3d %3d %12.4f' % (m, n,
                                np.sum(self.W[m] * self.W[n]) * self.dV))

    def symmetrize(self):
        r"""Symmetrize Hamiltonian.

        .. math::

            H_{\vec k} = H_{\vec k}^\dagger,
            H_{\vec R} = H_{-\vec R}^\dagger
        """
        if comm.rank == 0:
            status = misc.StatusBar(len(self.R),
                title='symmetrize Hamiltonian')

            for n in range(len(self.R)):
                N = misc.vector_index(self.R, -self.R[n])

                if N is None:
                    self.data[n] = 0.0
                else:
                    self.data[n] += self.data[N].T.conj()
                    self.data[n] /= 2
                    self.data[N] = self.data[n].T.conj()

                status.update()

        comm.Bcast(self.data)

    def supercell(self, N1=1, N2=1, N3=1):
        """Map tight-binding model onto supercell.

        Parameters
        ----------
        N1, N2, N3 : tuple of int or int, default 1
            Supercell lattice vectors in units of primitive lattice vectors.
            Multiples of single primitive vector can be defined via a scalar
            integer, linear combinations via a 3-tuple of integers.

        Returns
        -------
        object
            Tight-binding model for supercell.
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

        el = Model()
        el.size = N * self.size
        el.cells = []
        el.N = [tuple(N1), tuple(N2), tuple(N3)]

        if comm.rank == 0:
            for n1 in range(N):
                for n2 in range(N):
                    for n3 in range(N):
                        indices = n1 * N1 + n2 * N2 + n3 * N3

                        if np.all(indices % N == 0):
                            el.cells.append(tuple(indices // N))

            assert len(el.cells) == N

            const = dict()

            status = misc.StatusBar(len(self.R),
                title='map hoppings onto supercell')

            for n in range(len(self.R)):
                for i, cell in enumerate(el.cells):
                    R = self.R[n] + np.array(cell)

                    R1, r1 = divmod(np.dot(R, B1), N)
                    R2, r2 = divmod(np.dot(R, B2), N)
                    R3, r3 = divmod(np.dot(R, B3), N)

                    R = R1, R2, R3

                    indices = r1 * N1 + r2 * N2 + r3 * N3
                    j = el.cells.index(tuple(indices // N))

                    A = i * self.size
                    B = j * self.size

                    if R not in const:
                        const[R] = np.zeros((el.size, el.size), dtype=complex)

                    const[R][A:A + self.size, B:B + self.size] = self.data[n]

                status.update()

            el.R = np.array(list(const.keys()), dtype=int)
            el.data = np.array(list(const.values()))

            count = len(const)
            const.clear()
        else:
            count = None

        count = comm.bcast(count)

        if comm.rank != 0:
            el.R = np.empty((count, 3), dtype=int)
            el.data = np.empty((count, el.size, el.size), dtype=complex)

        comm.Bcast(el.R)
        comm.Bcast(el.data)

        el.cells = comm.bcast(el.cells)

        return el

    def unit_cell(self):
        """Map tight-binding model back to unit cell.

        Original idea by Bin Shao.

        See Also
        --------
        supercell
        """
        el = Model()
        el.size = self.size // len(self.cells)

        if comm.rank == 0:
            const = dict()

            status = misc.StatusBar(len(self.R),
                title='map hoppings back to unit cell')

            for n in range(len(self.R)):
                for i, cell in enumerate(self.cells):
                    t = self.data[n, :el.size, i * el.size:(i + 1) * el.size]

                    if np.any(t != 0):
                        R = tuple(np.dot(self.R[n], self.N) + np.array(cell))
                        const[R] = t

                status.update()

            el.R = np.array(list(const.keys()), dtype=int)
            el.data = np.array(list(const.values()))

            count = len(const)
            const.clear()
        else:
            count = None

        count = comm.bcast(count)

        if comm.rank != 0:
            el.R = np.empty((count, 3), dtype=int)
            el.data = np.empty((count, el.size, el.size), dtype=complex)

        comm.Bcast(el.R)
        comm.Bcast(el.data)

        return el

    def order_orbitals(self, *order):
        """Reorder Wannier functions.

        Warning: Wannier functions in position representation and related
        attributes remain unchanged!

        Together with :func:`shift_orbitals`, this function helps reconcile
        inconsistent definitions of the basis/motif of the Bravais lattice.

        Parameters
        ----------
        *order : int
            New order of Wannier functions.
        """
        self.data = self.data[:, order, :]
        self.data = self.data[:, :, order]

    def shift_orbitals(self, s, S):
        """Move selected Wannier functions across unit-cell boundary.

        Warning: Wannier functions in position representation and related
        attributes remain unchanged!

        Together with :func:`order_orbitals`, this function helps reconcile
        inconsistent definitions of the basis/motif of the Bravais lattice.

        Parameters
        ----------
        s : slice
            Slice of orbital indices corresponding to selected basis atom(s).
        S : tuple of int
            Shift of as multiple of primitive lattice vectors.
        """
        S = np.asarray(S)
        data = self.data.copy()

        old_R = set(range(len(self.R)))
        new_R = []
        new_t = []

        for i in range(len(self.R)):
            R = self.R[i] + S
            j = misc.vector_index(self.R, R)

            if j is None:
                t = np.zeros((self.size, self.size), dtype=complex)
                t[:, s] = self.data[i, :, s]
                t[s, s] = 0.0
                new_R.append(R)
                new_t.append(t)
            else:
                old_R.remove(j)
                data[j, :, s] = self.data[i, :, s]
                data[j, s, s] = self.data[j, s, s]

        for j in old_R:
            data[j, :, s] = 0.0
            data[j, s, s] = self.data[j, s, s]

        self.R = np.concatenate((self.R, new_R))
        self.data = np.concatenate((data, new_t))

        self.standardize()

    def standardize(self):
        """Standardize tight-binding data.

        - Keep only nonzero hopping matrices.
        - Sum over repeated lattice vectors.
        - Sort lattice vectors.
        """
        if comm.rank == 0:
            const = dict()

            for n in range(len(self.R)):
                if np.any(self.data[n] != 0.0):
                    R = tuple(self.R[n])

                    if R in const:
                        const[R] += self.data[n]
                    else:
                        const[R] = self.data[n]

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

def read_hrdat(hrdat, divide_ndegen=True):
    """Read *_hr.dat* file from Wannier90."""

    if comm.rank == 0:
        data = open(hrdat)

        # read all words of current line:

        def cols():
            return data.readline().split()

        # skip header:

        date = cols()[2]

        # get dimensions:

        num_wann = int(cols()[0])
        nrpts = int(cols()[0])

        # read degeneracies of Wigner-Seitz grid points:

        ndegen = []

        while len(ndegen) < nrpts:
            ndegen.extend(map(float, cols()))

        if not divide_ndegen:
            ndegen = [1] * len(ndegen)
    else:
        num_wann = nrpts = None

    num_wann = comm.bcast(num_wann)
    nrpts = comm.bcast(nrpts)

    cells = np.empty((nrpts, 3), dtype=int)
    const = np.empty((nrpts, num_wann, num_wann), dtype=complex)

    if comm.rank == 0:
        # read lattice vectors and hopping constants:

        size = num_wann ** 2

        for n in range(nrpts):
            for _ in range(size):
                tmp = cols()

                const[n, int(tmp[3]) - 1, int(tmp[4]) - 1] = (
                    float(tmp[5]) + 1j * float(tmp[6])) / ndegen[n]

            cells[n] = list(map(int, tmp[:3]))

        data.close()

    comm.Bcast(cells)
    comm.Bcast(const)

    return cells, const

def read_wsvecdat(wsvecdat):
    """Read *_wsvec.dat* file from Wannier90."""

    supvecs = dict()

    if comm.rank == 0:
        try:
            with open(wsvecdat) as data:
                next(data)

                for line in data:
                    i, j, k, a, b = tuple(map(int, line.split()))

                    supvecs[i, j, k, a - 1, b - 1] = [list(map(int,
                        next(data).split())) for _ in range(int(next(data)))]

        except FileNotFoundError:
            supvecs = None
            print('Warning: File "%s" not found!' % wsvecdat)

    supvecs = comm.bcast(supvecs)

    return supvecs

def read_bands(filband):
    """Read bands from *filband* just like Quantum ESRESSO's ``plotband.x``."""

    if comm.rank == 0:
        data = open(filband)

        header = next(data)

        # &plot nbnd=  13, nks=  1296 /
        _, nbnd, nks = header.split('=')
        nbnd = int(nbnd[:nbnd.index(',')])
        nks = int(nks[:nks.index('/')])
    else:
        nbnd = nks = None

    nbnd = comm.bcast(nbnd)
    nks = comm.bcast(nks)

    k = np.empty((nks, 3))
    x = np.empty((nks))
    bands = np.empty((nbnd, nks))

    if comm.rank == 0:
        for ik in range(nks):
            k[ik] = list(map(float, next(data).split()))

            energies = []

            while len(energies) < nbnd:
                energies.extend(list(map(float, next(data).split())))

            bands[:, ik] = energies

        data.close()

        x[0] = 0.0

        for ik in range(1, nks):
            dk = k[ik] - k[ik - 1]
            x[ik] = x[ik - 1] + np.sqrt(np.dot(dk, dk))

    comm.Bcast(k)
    comm.Bcast(x)
    comm.Bcast(bands)

    return k, x, bands

def read_bands_plot(filbandgnu, bands):
    """Read bands from *filband.gnu* produced by Quantum ESPRESSO's ``bands.x``.

    Parameters
    ----------
    filbandgnu : str
        Name of file with plotted bands.
    bands : int
        Number of bands.

    Returns
    -------
    ndarray
        Cumulative reciprocal distance.
    ndarray
        Band energies.
    """
    k, e = np.loadtxt(filbandgnu).T

    points = k.size // bands

    k = k[:points]
    e = np.reshape(e, (bands, points)).T

    return k, e

def read_symmetry_points(bandsout):
    """Read positions of symmetry points along path.

    Parameters
    ----------
    bandsout : str
        File with standard output from Quantum ESPRESSO's ``bands.x``.

    Returns
    -------
    list
        Positions of symmetry points.
    """
    points = []

    with open(bandsout) as data:
        for line in data:
            if 'x coordinate' in line:
                points.append(float(line.split()[-1]))

    return points

def read_atomic_projections(atomic_proj_xml, order=False, from_fermi=True,
        other=False, **order_kwargs):
    """Read projected bands from *outdir/prefix.save/atomic_proj.xml*.

    Parameters
    ----------
    atomic_proj_xml : str
        XML file with atomic projections generated by ``projwfc.x``.
    order : bool
        Order/disentangle bands via their k-local character?
    from_fermi : bool
        Subtract Fermi level from electronic energies?
    other : bool
        Estimate projection onto "other" orbitals as difference of band weights
        to one?

    Returns
    -------
    ndarray
        k points.
    ndarray
        Cumulative path distance.
    ndarray
        Electronic energies.
    ndarray
        Projections onto (pseudo) atomic orbitals.
    """
    if comm.rank == 0:
        data = open(atomic_proj_xml)
        next(data)

        header = next(data).strip('<HEADER />\n')
        header = dict([item.split('=') for item in header.split(' ')])
        header = dict((key, value.strip('"')) for key, value in header.items())

        bands = int(header['NUMBER_OF_BANDS'])
        nk = int(header['NUMBER_OF_K-POINTS'])
        no = int(header['NUMBER_OF_ATOMIC_WFC'])
        mu = float(header['FERMI_ENERGY'])
    else:
        bands = nk = no = None

    bands = comm.bcast(bands)
    nk = comm.bcast(nk)
    no = comm.bcast(no)

    x = np.empty(nk)
    k = np.empty((nk, 3))
    eps = np.empty((nk, bands))
    proj2 = np.empty((nk, bands, no + 1 if other else no))

    if comm.rank == 0:
        proj = np.empty((nk, bands, no), dtype=complex)

        next(data)

        for ik in range(nk):
            next(data) # <K-POINT>
            k[ik] = list(map(float, next(data).split()))
            next(data) # </K-POINT>

            next(data) # <E>
            levels = []
            while len(levels) < bands:
                levels.extend(list(map(float, next(data).split())))
            eps[ik] = levels
            next(data) # </E>

            next(data) # <PROJS>
            for a in range(no):
                next(data) # <ATOMIC_WFC>
                for n in range(bands):
                    Re, Im = list(map(float, next(data).split()))
                    proj[ik, n, a] = Re + 1j * Im
                next(data) # </ATOMIC_WFC>
            next(data) # </PROJS>

        data.close()

        x[0] = 0

        for i in range(1, nk):
            dk = k[i] - k[i - 1]
            x[i] = x[i - 1] + np.sqrt(np.dot(dk, dk))

        if from_fermi:
            eps -= mu

        if order:
            o = dispersion.band_order(eps,
                np.transpose(proj, axes=(0, 2, 1)).copy(), **order_kwargs)

            for ik in range(nk):
                eps[ik] = eps[ik, o[ik]]

                for a in range(no):
                    proj[ik, :, a] = proj[ik, o[ik], a]

        proj2[:, :, :no] = abs(proj) ** 2

        if other:
            proj2[:, :, no] = 1.0 - proj2[:, :, :no].sum(axis=2)

    comm.Bcast(x)
    comm.Bcast(k)
    comm.Bcast(eps)
    comm.Bcast(proj2)

    return x, k, eps, proj2

def read_atomic_projections_old(atomic_proj_xml, order=False, from_fermi=True,
        **order_kwargs):
    """Read projected bands from *outdir/prefix.save/atomic_proj.xml*."""

    if comm.rank == 0:
        data = open(atomic_proj_xml)

        def goto(pattern):
            for line in data:
                if pattern in line:
                    return line

        goto('<NUMBER_OF_BANDS ')
        bands = int(next(data))

        goto('<NUMBER_OF_K-POINTS ')
        nk = int(next(data))

        goto('<NUMBER_OF_ATOMIC_WFC ')
        no = int(next(data))

        goto('<FERMI_ENERGY ')
        mu = float(next(data))
    else:
        bands = nk = no = None

    bands = comm.bcast(bands)
    nk = comm.bcast(nk)
    no = comm.bcast(no)

    x = np.empty(nk)
    k = np.empty((nk, 3))
    eps = np.empty((nk, bands))
    proj = np.empty((nk, bands, no), dtype=complex)

    if comm.rank == 0:
        goto('<K-POINTS ')
        for i in range(nk):
            k[i] = list(map(float, next(data).split()))

        for ik in range(nk):
            goto('<EIG ')
            for n in range(bands):
                eps[ik, n] = float(next(data))

        for ik in range(nk):
            for a in range(no):
                goto('<ATMWFC.')
                for n in range(bands):
                    Re, Im = list(map(float, next(data).split(',')))
                    proj[ik, n, a] = Re + 1j * Im

        data.close()

        x[0] = 0

        for i in range(1, nk):
            dk = k[i] - k[i - 1]
            x[i] = x[i - 1] + np.sqrt(np.dot(dk, dk))

        if from_fermi:
            eps -= mu

        if order:
            o = dispersion.band_order(eps,
                np.transpose(proj, axes=(0, 2, 1)).copy(), **order_kwargs)

            for ik in range(nk):
                eps[ik] = eps[ik, o[ik]]

                for a in range(no):
                    proj[ik, :, a] = proj[ik, o[ik], a]

    comm.Bcast(x)
    comm.Bcast(k)
    comm.Bcast(eps)
    comm.Bcast(proj)

    return x, k, eps, abs(proj) ** 2

def read_projwfc_out(projwfc_out):
    """Identify orbitals in *atomic_proj.xml* via output of ``projwfc.x``

    Parameters
    ----------
    projwfc_out : str
        Output file of ``projwfc.x``.

    Returns
    -------
    list of str
        Common names of (pseudo) atomic orbitals listed in `projwfc_out` (in
        that order).
    """
    if comm.rank == 0:
        orbitals = []

        labels = [
            ['s'],
            ['pz', 'px', 'py'],
            ['dz2', 'dxz', 'dyz', 'dx2-y2', 'dxy'],
            ['fz3', 'fxz2', 'fyz2', 'fz(x2-y2)', 'fxyz', 'fx(x2-3y2)',
                'fy(3x2-y2)'],
            ]

        with open(projwfc_out) as data:
            for line in data:
                if 'Atomic states used for projection' in line:
                    next(data)
                    next(data)

                    while True:
                        info = next(data, '').rstrip()

                        if info:
                            X = info[28:31].strip()
                            n = int(info[39])
                            l = int(info[44])
                            m = int(info[49])

                            orbitals.append('%s-%d%s'
                                % (X, n, labels[l][m - 1]))
                        else:
                            break

            for orbital in set(orbitals):
                if orbitals.count(orbital) > 1:
                    duplicates = [n for n in range(len(orbitals))
                        if orbital == orbitals[n]]

                    for m, n in enumerate(duplicates, 1):
                        orbitals[n] = orbitals[n].replace('-', '%d-' % m, 1)

            orbitals.append('X-x')
    else:
        orbitals = None

    orbitals = comm.bcast(orbitals)

    return orbitals

def proj_sum(proj, orbitals, *groups, **kwargs):
    """Sum over selected atomic projections.

    Example:

    .. code-block:: python

        proj = read_atomic_projections('atomic_proj.xml')
        orbitals = read_projwf_out('projwfc.out')
        proj = proj_sum(proj, orbitals, 'S-p', 'Ta-d{z2, x2-y2, xy}')

    Parameters
    ----------
    proj : ndarray
        Atomic projections from :func:`read_atomic_projections`.
    orbitals : list of str
        Names of all orbitals from :func:`read_projwfc_out`.
    *groups
        Comma-separated lists of names of selected orbitals. Omitted name
        components are summed over. Curly braces are expanded.
    other : bool, default False
        Return remaining orbital weight too?

    Returns
    -------
    ndarray
        Summed-over atomic projections.
    """
    import re

    def info(orbital):
        return re.match('(?:([A-Z][a-z]?)(\d*))?-?(\d*)(?:([spdfx])(\S*))?',
            orbital.strip()).groups()

    other = kwargs.get('other', False)

    summed = np.empty(proj.shape[:2] + (len(groups) + 1 if other
        else len(groups),))

    if comm.rank == 0:
        orbitals = list(map(info, orbitals))

        for n, group in enumerate(groups):
            indices = set()

            for selection in map(info, misc.split(group)):
                for a, orbital in enumerate(orbitals):
                    if all(A == B for A, B in zip(selection, orbital) if A):
                        indices.add(a)

            summed[..., n] = proj[..., sorted(indices)].sum(axis=2)

        if other:
            summed[..., -1] = proj.sum(axis=2) - summed[..., :-1].sum(axis=2)

    comm.Bcast(summed)

    return summed

def read_Fermi_level(pw_scf_out):
    """Read Fermi level from output of self-consistent PW run."""

    if comm.rank == 0:
        with open(pw_scf_out) as data:
            for line in data:
                if 'Fermi energy' in line:
                    eF = float(line.split()[-2])
                elif 'highest occupied level' in line:
                    eF = float(line.split()[-1])
    else:
        eF = None

    eF = comm.bcast(eF)

    return eF

def read_pwo(pw_scf_out):
    """Read energies from PW output file."""

    Ne = Ns = Nk = E = None
    e_given = False

    if comm.rank == 0:
        with open(pw_scf_out) as lines:
            for line in lines:
                if 'number of electrons' in line:
                    Ne = float(line.split()[-1])

                elif 'number of Kohn-Sham states' in line:
                    Ns = int(line.split()[-1])

                elif 'number of k points' in line:
                    Nk = int(line.split()[4])

                elif 'Fermi energy' in line:
                    eF = float(line.split()[-2])

                elif 'highest occupied, lowest unoccupied level' in line:
                    eF = sum(map(float, line.split()[-2:])) / 2

                elif line.startswith('!'):
                    E = float(line.split()[-2]) * misc.Ry

                elif 'End of self-consistent calculation' in line:
                    try:
                        e = []

                        for ik in range(Nk):
                            for _ in range(3):
                                next(lines)

                            e.append([])

                            while len(e[-1]) < Ns:
                                e[-1].extend(list(map(float,
                                    next(lines).split())))

                        e_given = True

                    except ValueError:
                        pass

        if e_given and eF is not None:
            e = np.array(e) - eF

    Ne = comm.bcast(Ne)
    Ns = comm.bcast(Ns)
    Nk = comm.bcast(Nk)
    E = comm.bcast(E)

    if comm.bcast(e_given):
        if comm.rank != 0:
            e = np.empty((Nk, Ns))

        comm.Bcast(e)
    else:
        e = None

    return e, Ne, E

def read_wannier90_eig_file(seedname, num_bands, nkpts):
    """Read Kohn-Sham energies (eV) from the Wannier90 output seedname.eig file.

    Parameters
    ----------
    seedname : str
        For example 'tas2', if the file is named 'tas2.eig'.
    num_bands : int
        Number of bands in your pseudopotential.
    nkpts : int
        Number of k-points in your Wannier90 calculations.
        For example 1296 for 36x36x1.

    Returns
    -------
    ndarray
        Kohn-Sham energies: ``eig[num_bands, nkpts]``.

    """
    eig = np.empty((num_bands, nkpts))

    f = open(seedname + '.eig', 'r')
    lines = f.readlines()

    for lineI in range(len(lines)):
        bandI, kI, eigI = lines[lineI].split()

        eig[int(bandI) - 1, int(kI) - 1] = np.real(eigI)

    f.close()

    return eig

def eband_from_qe_pwo(pw_scf_out, subset=None):
    """Calculate ``eband`` part of one-electron energy.

    The 'one-electron contribution' energy in the Quantum ESPRESSO PWscf output
    is a sum ``eband + deband``. Here, we can calculate the ``eband`` part.

    To compare it with the Quantum ESPRESSO result, you need to modify
    the ``SUBROUTINE print_energies ( printout )`` from *electrons.f90*.

    Change::

        WRITE( stdout, 9060 ) &
            ( eband + deband ), ehart, ( etxc - etxcc ), ewld

    to::

        WRITE( stdout, 9060 ) &
            eband, ( eband + deband ), ehart, ( etxc - etxcc ), ewld

    and::

        9060 FORMAT(/'     The total energy is the sum of the following terms:',/,&
                /'     one-electron contribution =',F17.8,' Ry' &

    to::

        9060 FORMAT(/'     The total energy is the sum of the following terms:',/,&
                /'     sum bands                 =',F17.8,' Ry' &
                /'     one-electron contribution =',F17.8,' Ry' &

    At some point, we should add the ``deband`` routine as well...

    Parameters
    ----------
    pw_scf_out : str
        The name of the output file (typically 'pw.out').
    subset : list or array
        List of indices to pick only a subset of the bands
        for the integration

    Returns
    -------
    eband : float
        The band energy.
    """
    f = open(pw_scf_out, 'r')

    lines = f.readlines()

    # read number of k points and smearing:

    for ii in np.arange(len(lines)):
        if lines[ii].find('     number of k points=') == 0:

            line_index = ii

    smearing_line = lines[line_index].split()
    N_k = int(smearing_line[4])
    kT = float(smearing_line[9])

    k_Points = np.empty([N_k, 4])

    for ii in np.arange(N_k):
        (kb, einsb, eq, bra, kx, ky, kz, wk_s, eq2,
            wk) = lines[line_index + 2 + ii].split()

        kx = float(kx)
        ky = float(ky)
        kz = float(kz[:-2])

        wk = float(wk)

        k_Points[ii, 0] = kx
        k_Points[ii, 1] = ky
        k_Points[ii, 2] = kz
        k_Points[ii, 3] = wk

    # read number of Kohn-Sham states:

    for ii in np.arange(len(lines)):
        if lines[ii].find('     number of Kohn-Sham') == 0:
            KS_index = ii

    number, of, KS_s, states, N_states = lines[KS_index].split()

    N_states = int(N_states)

    # read all energies for all the different k points and Kohn-Sham States:

    for ii in np.arange(len(lines)):
        if lines[ii].find('     End of') == 0:
            state_start_index = ii + 4

    energies = np.zeros((N_k, N_states))

    # the states are written in columns of size 8
    # with divmod we check how many rows we have
    state_lines = divmod(N_states, 8)[0]
    if divmod(N_states, 8)[1] != 0:
        state_lines += 1

    for ik in np.arange(N_k):
        energies_per_k = []
        for istate in range(state_lines):
            energies_per_k.extend(lines[state_start_index
                + istate + (state_lines + 3) * ik].split())

        energies[ik] = np.array(energies_per_k)

    kT *= misc.Ry

    mu = read_Fermi_level(pw_scf_out)

    eband = np.zeros(energies.shape)

    if subset == None:
        for ik in range(N_k):
            for iband in range(N_states):
                eband[ik, iband] = (energies[ik, iband] * k_Points[ik, 3]
                    * occupations.fermi_dirac((energies[ik, iband] - mu) / kT))
    else:
        for ik in range(N_k):
            for iband in subset:
                eband[ik, iband] = (energies[ik, iband] * k_Points[ik, 3]
                    * occupations.fermi_dirac((energies[ik, iband] - mu) / kT))

    eband = eband.sum() / misc.Ry

    return eband

def read_decayH(file):
    """Read *decay.H* output from EPW.

    Parameters
    ----------
    file : str
        The name of the *decay.H* output from EPW.

    Returns
    -------
    R : ndarray
        The distance of every Wigner-Seitz grid point measured from the center
        in angstrom.
    H : ndarray
        The maximum absolute value of the elements of the Hamiltonian matrix in
        rydberg.
    """
    with open(file) as f_decay:
        lines = f_decay.readlines()

    R_list = []
    H_list = []

    # read in lines with 2 entries:
    for line in range(len(lines)):
        if len(lines[line].split()) == 2:
            R_str, H_str = lines[line].split()
            R_list.append(float(R_str))
            H_list.append(float(H_str))

    # convert list to NumPy array:
    R = np.asarray(R_list)
    H = np.asarray(H_list)

    return R, H

def decayH(seedname, **kwargs):
    """Calculate the decay of the Hamiltonian.

    This function should yield the same data as :func:`read_decayH`.

    Parameters
    ----------
    seedname : str
        Prefix of Wannier90 output file.

    **kwargs
        Arguments for :func:`bravais.primitives`: Choose the right Bravais
        lattice (``ibrav``) and lattice constants (``a, b, c, ...``).

        For a simple cubic lattice: ``decayH(seedname, ibrav=1, a=a)``.

    Returns
    -------
    R : ndarray
        The distance of every Wigner-Seitz grid point measured from the center
        in angstrom.
    H : ndarray
        The maximum absolute value of the elements of the Hamiltonian matrix in
        rydberg.
    """
    bravais_vectors = bravais.primitives(**kwargs)
    el = Model(seedname, divide_ndegen=False)

    R = np.empty((len(el.R)))
    H = np.empty((len(el.R)))

    # loop over all Wigner-seitz grid points
    for ii in range(len(el.R)):

        distance = np.empty((3, 3))
        for xi in range(3):
            distance[xi, :] = el.R[ii][xi] * bravais_vectors[xi, :]
        distance = distance.sum(axis=0)

        R[ii] = np.linalg.norm(distance)
        H[ii] = np.max(abs(el.data[ii])) / misc.Ry

    return R, H

def read_energy_contributions_scf_out(filename):
    """Read energy contributions to the total energy
    from Quantum ESPRESSO's scf output file.

    Parameters
    ----------
    filename : str
        scf output file name.

    Returns
    -------
    dict
        energy contributions.
    """
    if comm.rank == 0:
        energies = dict()

        with open(filename) as lines:
            for line in lines:
                words = [word
                    for column in line.split()
                    for word in column.split('=') if word]

                if not words:
                    continue

                key = words[0].lower()

                if key in 'sum bands':
                    if words[0] == 'sum':
                        energies['sum bands'] = words[2]
                elif key in 'one-electron contribution':
                    if words[0] == 'one-electron':
                        energies[key] = words[2]
                elif key in 'hartree contribution':
                    if words[0] == 'hartree':
                        energies[key] = words[2]
                elif key in 'xc contribution':
                    if words[0] == 'xc':
                        energies[key] = words[2]
                elif key in 'ewald contribution':
                    if words[0] == 'ewald':
                        energies[key] = words[2]
                elif key in 'smearing contrib.':
                    if words[0] == 'smearing':
                        energies[key] = words[3]
                elif key in '!':
                    if words[0] == '!':
                        energies['total'] = words[3]

    else:
        energies = None

    energies = comm.bcast(energies)

    return energies
