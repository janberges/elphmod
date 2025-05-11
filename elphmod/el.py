# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

"""Tight-binding models from Wannier90."""

import numpy as np

import elphmod.bravais
import elphmod.dispersion
import elphmod.misc
import elphmod.MPI
import elphmod.occupations

comm = elphmod.MPI.comm
info = elphmod.MPI.info

class Model:
    r"""Tight-binding model for the electrons.

    Parameters
    ----------
    seedname : str
        Common prefix of Wannier90 output files: *seedname_hr.dat* with the
        Hamiltonian in the Wannier basis and, optionally, *seedname_wsvec.dat*
        with superlattice vectors used to symmetrize the long-range hopping.
        Alternatively, *dat.h_mat_r* from RESPACK can be used.
    N : tuple of int, optional
        Numbers of unit cells per direction on which RESPACK data is defined.
        This can be omitted if all numbers are even.
    a : ndarray, optional
        Bravais lattice vectors used to map RESPACK data to Wigner-Seitz cell.
        By default, a cubic cell is assumed.
    r : ndarray, optional
        Positions of orbital centers used to map RESPACK data to Wigner-Seitz
        cell. By default, all orbitals are assumed to be located at the origin
        of the unit cell.
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
    rydberg : bool, default False
        Convert energies from eV to Ry?
    shared_memory : bool, default False
        Store Wannier functions in shared memory?

    Attributes
    ----------
    R : ndarray
        Lattice vectors :math:`\vec R` of Wigner-Seitz supercell.
    data : ndarray
        Corresponding on-site energies and hoppings in eV.

        .. math::

            H_{\vec R \alpha \beta} = \bra{0 \alpha} H \ket{\vec R \beta}

        If :attr:`rydberg` is ``True``, the units are Ry instead.
    size : int
        Number of Wannier functions/bands.
    nk : tuple of int
        Guessed shape of original k-point mesh.
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
    divide_ndegen : bool
        Have hoppings been divided by degeneracy of Wigner-Seitz point?
    rydberg : bool
        Have energies been converted from eV to Ry?
    """
    def H(self, k1=0, k2=0, k3=0):
        r"""Set up Hamilton operator for arbitrary k point.

        Parameters
        ----------
        k1, k2, k3 : float, default 0.0
            k point in crystal coordinates with period :math:`2 \pi`.

        Returns
        -------
        ndarray
            Fourier transform of :attr:`data`.
        """
        k = np.array([k1, k2, k3])

        # Sign convention in hamiltonian.f90 of Wannier90:
        # 300  fac = exp(-cmplx_i*rdotk)/real(num_kpts, dp)
        # 301  ham_r(:, :, irpt) = ham_r(:, :, irpt) + fac*ham_k(:, :, loop_kpt)

        # Note that the data from Wannier90 can be interpreted like this:
        # self.data[self.R == R - R', a, b] = <R' a|H|R b>

        # Compare this convention [doi:10.26092/elib/250, Eq. 2.35a]:
        # t(R - R', a, b) = <R a|H|R' b>

        return np.einsum('Rab,R->ab', self.data, np.exp(1j * self.R.dot(k)))

    def v(self, k1=0, k2=0, k3=0):
        r"""Set up band-velocity operator for arbitrary k point.

        Parameters
        ----------
        k1, k2, k3 : float, default 0.0
            k point in crystal coordinates with period :math:`2 \pi`.

        Returns
        -------
        ndarray
            k derivative of Fourier transform of :attr:`data` in the basis of
            primitive lattice vectors and orbitals. It can be transformed into
            the Cartesian and band basis afterward (Hellmann-Feynman theorem).
        """
        k = np.array([k1, k2, k3])

        return np.einsum('Rx,Rab,R->xab',
            1j * self.R, self.data, np.exp(1j * self.R.dot(k)))

    def t(self, R1=0, R2=0, R3=0):
        """Get on-site or hopping energy for arbitrary lattice vector.

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

    def __init__(self, seedname=None, N=None, a=None, r=None,
            divide_ndegen=True, read_xsf=False, normalize_wf=False,
            buffer_wf=False, check_ortho=False, rydberg=False,
            shared_memory=False):

        self.divide_ndegen = divide_ndegen
        self.rydberg = rydberg

        self.cells = [(0, 0, 0)]

        if seedname is None:
            return

        if seedname.endswith('_hr.dat'):
            seedname = seedname[:-7]

        if seedname.endswith('dat.h_mat_r'):
            R, data = elphmod.misc.read_dat_mat(seedname)
            self.size = data.shape[1]

            if N is None:
                N = 2 * R[-1]

            if a is None:
                info('Warning: You should really define the Bravais lattice!')
                a = elphmod.bravais.primitives(ibrav=1)

            if r is None:
                r = np.zeros((self.size, 3))

            t = np.zeros((N[0], N[1], N[2], self.size, self.size),
                dtype=complex)

            for iR, (R1, R2, R3) in enumerate(R):
                t[R1 % N[0], R2 % N[1], R3 % N[2]] = data[iR]

            k2r(self, t, a, r, fft=False)

            supvecs = None
        else:
            self.R, self.data = read_hrdat(seedname, divide_ndegen)
            self.size = self.data.shape[1]

            if rydberg:
                self.data /= elphmod.misc.Ry

            supvecs = read_wsvecdat('%s_wsvec.dat' % seedname)

        self.nk = tuple(2 * self.R[np.all(self.R[:, x] == 0,
            axis=1)].max(initial=1) for x in [[1, 2], [2, 0], [0, 1]])

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
                self.W = elphmod.MPI.load('%s_wf.npy' % seedname,
                    shared_memory)

                self.r = elphmod.MPI.load('%s_xyz.npy' % seedname,
                    shared_memory)

            read_buffer = buffer_wf and self.W.size and self.r.size

            r0, a, self.atom_order, self.tau, shape = elphmod.misc.read_xsf(
                '%s_head.xsf' % seedname if read_buffer else
                '%s_%05d.xsf' % (seedname, 1), only_header=True)

            self.dV = abs(np.dot(np.cross(a[0], a[1]), a[2])) / np.prod(shape)

            if not read_buffer:
                self.r = elphmod.misc.real_space_grid(shape, r0, a,
                    shared_memory)

                sizes, bounds = elphmod.MPI.distribute(self.size, bounds=True)

                my_W = np.empty((sizes[comm.rank], *shape))
                node, images, self.W = elphmod.MPI.shared_array(
                    (self.size, *shape), shared_memory=shared_memory)

                for my_n, n in enumerate(range(
                        *bounds[comm.rank:comm.rank + 2])):

                    my_W[my_n] = elphmod.misc.read_xsf('%s_%05d.xsf'
                        % (seedname, n + 1), comm=elphmod.MPI.I)[-1]

                    if normalize_wf:
                        my_W[my_n] /= np.sqrt(np.sum(my_W[my_n] ** 2) * self.dV)

                comm.Gatherv(my_W, (self.W, sizes * np.prod(shape)))

                if node.rank == 0:
                    images.Bcast(self.W)

                comm.Barrier()

                if buffer_wf and comm.rank == 0:
                    elphmod.misc.write_xsf('%s_head.xsf' % seedname, r0, a,
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

    def supercell(self, N1=1, N2=1, N3=1, sparse=False):
        """Map tight-binding model onto supercell.

        Parameters
        ----------
        N1, N2, N3 : tuple of int or int, default 1
            Supercell lattice vectors in units of primitive lattice vectors.
        sparse : bool, default False
            Only calculate k = 0 Hamiltonian as a sparse matrix to save memory?
            The result, which is assumed to be real, is stored in the attribute
            :attr:`Hs`. Consider using :meth:`standardize` with nonzero `eps`
            and `symmetrize` before.

        Returns
        -------
        object
            Tight-binding model for supercell.

        See Also
        --------
        elphmod.bravais.supercell
        """
        el = Model()

        supercell = elphmod.bravais.supercell(N1, N2, N3)
        el.N = list(map(tuple, supercell[1]))
        el.cells = supercell[-1]

        el.size = len(el.cells) * self.size

        el.divide_ndegen = self.divide_ndegen
        el.rydberg = self.rydberg

        if sparse:
            sparse_array = elphmod.misc.get_sparse_array()

            el.Hs = sparse_array((el.size, el.size))

            if abs(self.data.imag).sum() > 1e-6 * abs(self.data.real).sum():
                info('Warning: Significant imaginary part of hopping ignored!')

        if comm.rank == 0:
            const = dict()

            status = elphmod.misc.StatusBar(len(el.cells),
                title='map hoppings onto supercell')

            for i in range(len(el.cells)):
                A = i * self.size

                for n in range(len(self.R)):
                    R, r = elphmod.bravais.to_supercell(self.R[n] + el.cells[i],
                        supercell)

                    B = r * self.size

                    if sparse:
                        el.Hs[
                            A:A + self.size,
                            B:B + self.size] += self.data[n].real
                        continue

                    if R not in const:
                        const[R] = np.zeros((el.size, el.size), dtype=complex)

                    const[R][A:A + self.size, B:B + self.size] = self.data[n]

                status.update()

            el.R = np.array(list(const.keys()), dtype=int).reshape((-1, 3))
            el.data = np.array(list(const.values())).reshape((-1,
                el.size, el.size))

            count = len(const)
            const.clear()
        else:
            count = None

        if sparse:
            el.Hs = comm.bcast(el.Hs)

            return el

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

            status = elphmod.misc.StatusBar(len(self.R),
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
        R = np.tile(self.R, (3, 1))
        data = np.tile(self.data, (3, 1, 1))

        n = len(self.R)
        m = 2 * n

        R[:n] -= S
        data[:n, :, :] = 0.0
        data[:n, :, s] = self.data[:, :, s]
        data[:n, s, s] = 0.0

        data[n:m, s, :] = 0.0
        data[n:m, :, s] = 0.0
        data[n:m, s, s] = self.data[:, s, s]

        R[m:] += S
        data[m:, :, :] = 0.0
        data[m:, s, :] = self.data[:, s, :]
        data[m:, s, s] = 0.0

        self.R = R
        self.data = data

        self.standardize()

    def standardize(self, eps=0.0, symmetrize=False):
        r"""Standardize tight-binding data.

        - Keep only nonzero hopping matrices.
        - Sum over repeated lattice vectors.
        - Sort lattice vectors.
        - Optionally symmetrize hopping:

        .. math::

            H_{\vec k} = H_{\vec k}^\dagger,
            H_{\vec R} = H_{-\vec R}^\dagger

        Parameters
        ----------
        eps : float
            Threshold for "nonzero" matrix elements in units of the maximum
            matrix element.
        symmetrize : bool
            Symmetrize hopping?
        """
        if comm.rank == 0:
            if eps:
                self.data[abs(self.data) < eps * abs(self.data).max()] = 0.0

            const = dict()

            status = elphmod.misc.StatusBar(len(self.R),
                title='standardize tight-binding data')

            for n in range(len(self.R)):
                if np.any(self.data[n] != 0.0):
                    R = tuple(self.R[n])

                    if R in const:
                        const[R] += self.data[n]
                    else:
                        const[R] = self.data[n].copy()

                    if symmetrize:
                        R = tuple(-self.R[n])

                        if R in const:
                            const[R] += self.data[n].T.conj()
                        else:
                            const[R] = self.data[n].T.conj()

                status.update()

            cells = sorted(list(const.keys()))
            count = len(cells)

            self.R = np.array(cells, dtype=int)
            self.data = np.array([const[R] for R in cells])

            if symmetrize:
                self.data /= 2
        else:
            count = None

        count = comm.bcast(count)

        if comm.rank != 0:
            self.R = np.empty((count, 3), dtype=int)
            self.data = np.empty((count, self.size, self.size), dtype=complex)

        comm.Bcast(self.R)
        comm.Bcast(self.data)

    def symmetrize(self):
        """Symmetrize Hamiltonian."""

        self.standardize(symmetrize=True)

    def clear(self):
        """Delete all lattice vectors and associated matrix elements."""

        self.R = np.zeros_like(self.R[:0, :])
        self.data = np.zeros_like(self.data[:0, :, :])

    def to_hrdat(self, seedname):
        """Save tight-binding model to *_hr.dat* file.

        Parameters
        ----------
        seedname : str
            Common prefix of Wannier90 input and output files.
        """
        if comm.rank == 0:
            write_hrdat(seedname, self.R,
                self.data * elphmod.misc.Ry if self.rydberg else self.data)

def read_hrdat(seedname, divide_ndegen=True):
    """Read *_hr.dat* (or *_tb.dat*) file from Wannier90.

    Parameters
    ----------
    seedname : str
        Common prefix of Wannier90 input and output files.
    divide_ndegen : bool
        Divide hopping by degeneracy of Wigner-Seitz point?

    Returns
    -------
    ndarray
        Lattice vectors.
    ndarray
        On-site and hopping parameters.
    """
    if comm.rank == 0:
        try:
            data = open('%s_hr.dat' % seedname)
            tb = False
        except FileNotFoundError:
            print('Warning: Hamiltonian read from "%s_tb.dat"!' % seedname)

            data = open('%s_tb.dat' % seedname)
            tb = True

        # read all words of current line:

        def cols():
            for line in data:
                words = line.split()

                if words:
                    return words

        # skip header:

        date = cols()[2]

        if tb:
            for _ in range(3):
                a = cols()

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

            if tb:
                cells[n] = list(map(int, cols()))

            for _ in range(size):
                tmp = cols()

                const[n, int(tmp[-4]) - 1, int(tmp[-3]) - 1] = (
                    float(tmp[-2]) + 1j * float(tmp[-1])) / ndegen[n]

            if not tb:
                cells[n] = list(map(int, tmp[:3]))

        data.close()

    comm.Bcast(cells)
    comm.Bcast(const)

    return cells, const

def write_hrdat(seedname, R, H, ndegen=None):
    """Write *_hr.dat* file as generated by Wannier90.

    Parameters
    ----------
    seedname : str
        Common prefix of Wannier90 input and output files.
    R : ndarray
        Lattice vectors of Wigner-Seitz supercell.
    H : ndarray
        Corresponding on-site energies and hoppings.
    ndegen : list of int
        Degeneracies of Wigner-Seitz lattice vectors. This is just what is
        written to the file header, and it is not used to further modify `H`.
    """
    import time

    R_orig = R
    R = np.zeros((len(R), 3), dtype=int)
    R[:, :R_orig.shape[1]] = R_orig

    size = H.shape[-1]
    H = H.reshape((len(R), size, size))

    if ndegen is None:
        ndegen = np.ones(len(R))

    order = np.lexsort(R.T[::-1])

    with open('%s_hr.dat' % seedname, 'w') as hr:
        hr.write(time.strftime(' written on %d%b%Y at %H:%M:%S\n'))

        hr.write('%12d\n' % size)
        hr.write('%12d\n' % len(R))

        columns = 15

        for n, i in enumerate(order, 1):
            hr.write('%5d' % ndegen[i])

            if not n % columns or n == len(R):
                hr.write('\n')

        form = '%5d' * 5 + '%12.6f' * 2 + '\n'

        for i in order:
            for b in range(size):
                for a in range(size):
                    hr.write(form % (R[i, 0], R[i, 1], R[i, 2],
                        a + 1, b + 1, H[i, a, b].real, H[i, a, b].imag))

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

def k2r(el, H, a, r, fft=True, rydberg=False):
    """Interpolate Hamilontian matrices on uniform k-point mesh.

    Parameters
    ----------
    el : :class:`Model`
        Tight-binding model.
    H : ndarray
        Hamiltonian matrices on complete uniform k-point mesh.
    a : ndarray
        Bravais lattice vectors.
    r : ndarray
        Positions of orbital centers.
    fft : bool
        Perform Fourier transform? If ``False``, only the mapping to the
        Wigner-Seitz cell is performed.
    rydberg : bool, default False
        Is input Hamiltonian given in Ry rather than eV units? This is
        independent of ``el.rydberg``, which is always respected.
    """
    nk = H.shape[:-2]
    el.size = H.shape[-2]

    nk_orig = tuple(nk)
    nk = np.ones(3, dtype=int)
    nk[:len(nk_orig)] = nk_orig

    if fft:
        H = np.reshape(H, (*nk, el.size, el.size))
        t = np.fft.ifftn(H.conj(), axes=(0, 1, 2)).conj()
    else:
        t = H

    t = np.reshape(t, (*nk, el.size, 1, el.size, 1))
    t = np.transpose(t, (3, 5, 0, 1, 2, 4, 6))

    el.R, el.data, l = elphmod.bravais.short_range_model(t, a, r,
        sgn=+1, divide_ndegen=el.divide_ndegen)

    el.nk = tuple(nk)

    if rydberg != el.rydberg:
        if rydberg:
            el.data *= elphmod.misc.Ry
        else:
            el.data /= elphmod.misc.Ry

def read_bands(filband):
    """Read bands from *filband* just like Quantum ESRESSO's ``plotband.x``.

    Parameters
    ----------
    filband : str
        Filename.

    Returns
    -------
    k : ndarray
        k points in Cartesian coordiantes.
    bands : ndarray
        Band energies.
    """
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
            k[ik] = list(map(float,
                next(data).strip().rstrip('*').split()[:3]))

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

def write_bands(filband, k, bands, fmt='%8.3f', cols=10):
    """Write bands to *filband* just like Quantum ESPRESSO's ``plotband.x``.

    Parameters
    ----------
    filband : str
        Filename.
    k : ndarray
        k points in Cartesian coordiantes.
    bands : ndarray
        Band energies.
    fmt : str
        Format string for band energies.
    cols : int
        Number of band energies per line.
    """
    if comm.rank != 0:
        return

    with open(filband, 'w') as data:
        data.write(' &plot nbnd=%4d, nks=%6d /\n' % (len(bands), len(k)))

        for ik in range(len(k)):
            data.write('%20.6f %9.6f %9.6f\n' % tuple(k[ik]))

            for n, e in enumerate(bands[:, ik], 1):
                data.write(' ' + fmt % e)

                if not n % cols or n == len(bands):
                    data.write('\n')

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
        squared=True, other=False, **order_kwargs):
    """Read projected bands from *outdir/prefix.save/atomic_proj.xml*.

    Parameters
    ----------
    atomic_proj_xml : str
        XML file with atomic projections generated by ``projwfc.x``.
    order : bool
        Order/disentangle bands via their k-local character?
    from_fermi : bool
        Subtract Fermi level from electronic energies?
    squared : bool
        Return squared complex modulus of projection?
    other : bool
        Estimate projection onto "other" orbitals not defined in pseudopotential
        files as difference of band weights to one? This requires `squared`.
    **order_kwargs
        Keyword arguments passed to :func:`band_order`.

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

    if squared:
        proj2 = np.empty((nk, bands, no + 1 if other else no))

    if comm.rank == 0 or not squared:
        proj = np.empty((nk, bands, no), dtype=complex)

    if comm.rank == 0:
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
            o = elphmod.dispersion.band_order(eps,
                np.transpose(proj, axes=(0, 2, 1)).copy(), **order_kwargs)

            for ik in range(nk):
                eps[ik] = eps[ik, o[ik]]

                for a in range(no):
                    proj[ik, :, a] = proj[ik, o[ik], a]

        if squared:
            proj2[:, :, :no] = abs(proj) ** 2

            if other:
                proj2[:, :, no] = 1.0 - proj2[:, :, :no].sum(axis=2)

    comm.Bcast(x)
    comm.Bcast(k)
    comm.Bcast(eps)

    if squared:
        comm.Bcast(proj2)
    else:
        comm.Bcast(proj)

    return x, k, eps, proj2 if squared else proj

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
            o = elphmod.dispersion.band_order(eps,
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

def read_projwfc_out(projwfc_out, other=True):
    """Identify orbitals in *atomic_proj.xml* via output of ``projwfc.x``.

    Parameters
    ----------
    projwfc_out : str
        Output file of ``projwfc.x``.

    Returns
    -------
    list of str
        Common names of (pseudo) atomic orbitals listed in `projwfc_out` (in
        that order). If spin-orbit coupling is considered, symmetry labels
        related to the magnetic quantum number are omitted.
    other : bool, default True
        Add name ``X-x`` to list of orbitals, which corresponds to "other"
        orbitals from :func:`read_atomic_projections`?
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
                            n = int(info[38:40])
                            l = int(info[44])

                            if info[46] == 'm':
                                m = int(info[48:50])

                                orbitals.append('%s-%d%s'
                                    % (X, n, labels[l][m - 1]))
                            else:
                                orbitals.append('%s-%d%s'
                                    % (X, n, labels[l][0][0]))
                        else:
                            break

            for orbital in set(orbitals):
                if orbitals.count(orbital) > 1:
                    duplicates = [n for n in range(len(orbitals))
                        if orbital == orbitals[n]]

                    for m, n in enumerate(duplicates, 1):
                        orbitals[n] = orbitals[n].replace('-', '%d-' % m, 1)

            if other:
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
        Return remaining orbital weight too? If you just want to select the
        "other" orbitals from :func:`read_atomic_projections`, add the name
        ``x`` to `orbitals` instead.

    Returns
    -------
    ndarray
        Summed-over atomic projections.
    """
    import re

    def info(orbital):
        return re.match(r'(?:([A-Z][a-z]?)(\d*))?-?(\d*)(?:([spdfx])(\S*))?',
            orbital.strip()).groups()

    other = kwargs.get('other', False)

    summed = np.empty((*proj.shape[:2],
        len(groups) + 1 if other else len(groups)))

    if comm.rank == 0:
        orbitals = list(map(info, orbitals))

        for n, group in enumerate(groups):
            indices = set()

            for selection in map(info, elphmod.misc.split(group)):
                for a, orbital in enumerate(orbitals):
                    if all(A == B for A, B in zip(selection, orbital) if A):
                        indices.add(a)

            if not indices:
                print('Warning: "%s" does not match any orbital!' % group)

            summed[..., n] = proj[..., sorted(indices)].sum(axis=2)

        if other:
            summed[..., -1] = 1 - summed[..., :-1].sum(axis=2)

    comm.Bcast(summed)

    return summed

def read_Fermi_level(pw_scf_out):
    """Read Fermi level from output of self-consistent PW run.

    Parameters
    ----------
    pw_scf_out : str
        PWscf output file.

    Returns
    -------
    float
        Fermi level if found, ``None`` otherwise.
    """
    eF = None

    if comm.rank == 0:
        with open(pw_scf_out) as data:
            for line in data:
                if 'the Fermi energy is' in line:
                    eF = float(line.split()[-2])
                elif 'highest occupied level' in line:
                    eF = float(line.split()[-1])

    return comm.bcast(eF)

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

                elif 'the Fermi energy is' in line:
                    eF = float(line.split()[-2])

                elif 'highest occupied, lowest unoccupied level' in line:
                    eF = sum(map(float, line.split()[-2:])) / 2

                elif line.startswith('!'):
                    E = float(line.split()[-2]) * elphmod.misc.Ry

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
        Number of k points in your Wannier90 calculations.
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

def read_eps_nk_from_qe_pwo(pw_scf_out):
    """Read electronic eigenenergies and k points and calculate the occupations.

    Parameters
    ----------
    pw_scf_out : str
        Name of the output file (typically ``'pw.out'`` or ``'scf.out'``).

    Returns
    -------
    energies : ndarray
        Electronic eigenenergies (``eps_nk``) from ``scf`` output, shape ``(nk,
        nbnd)``.
    kpoints : ndarray
        k points and weights from ``scf`` output, shape ``(nk, 4)``, where the
        4th column contains the weights.
    f_occ : ndarray
        Occupations of ``eps_nk`` (same shape as `energies`).
    smearing_type : str
        Type of smearing (for example ``'Fermi-Dirac'``).
    mu : float
        Chemical potential in eV.
    kT : float
        Value of smearing (``degauss``) in eV.
    """
    scf_file = open(pw_scf_out, 'r')
    lines = scf_file.readlines()
    scf_file.close()

    # read number of k points and smearing:

    for ii in np.arange(len(lines)):
        if lines[ii].find('     number of k points=') == 0:

            line_index = ii

    smearing_line = lines[line_index].split()
    nk = int(smearing_line[4])
    smearing_type = smearing_line[5]
    f = elphmod.occupations.smearing(smearing_type)
    kT = float(smearing_line[9])

    k_Points = np.empty([nk, 4])

    for ii in np.arange(nk):
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

    nbnd = int(lines[KS_index].split()[4])

    # read all energies for all the different k points and Kohn-Sham States:

    for ii in np.arange(len(lines)):
        if lines[ii].find('     End of') == 0:
            state_start_index = ii + 4

    energies = np.zeros((nk, nbnd))
    f_occ = np.zeros((nk, nbnd))

    # the states are written in columns of size 8
    # with divmod we check how many rows we have
    state_lines = divmod(nbnd, 8)[0]
    if divmod(nbnd, 8)[1] != 0:
        state_lines += 1

    for ik in np.arange(nk):
        energies_per_k = []
        for istate in range(state_lines):
            energies_per_k.extend(lines[state_start_index
                + istate + (state_lines + 3) * ik].split())

        energies[ik] = np.array(energies_per_k)

    # kT: from Ry to eV
    kT *= elphmod.misc.Ry
    # read chemical potential
    mu = read_Fermi_level(pw_scf_out)
    # calculate occupations for all energies (eps_nk)
    f_occ = f((energies - mu) / kT)

    return energies, k_Points, f_occ, smearing_type, mu, kT

def eband(pw_scf_out, subset=None):
    """Calculate ``eband`` part of one-electron energy.

    The 'one-electron contribution' energy in the Quantum ESPRESSO PWscf output
    is a sum ``eband + deband``. Here, we can calculate the ``eband`` part.

    To compare it with the Quantum ESPRESSO result, you need to modify
    the ``SUBROUTINE print_energies ( printout )`` from *electrons.f90*.

    Change::

        WRITE( stdout, 9062 ) (eband + deband), ehart, ( etxc - etxcc ), ewld

    to::

        WRITE( stdout, 9062 ) eband, &
           (eband + deband), ehart, ( etxc - etxcc ), ewld

    and::

        9062 FORMAT( '     one-electron contribution =',F17.8,' Ry' &

    to::

        9062 FORMAT( '     sum bands                 =',F17.8,' Ry' &
                    /'     one-electron contribution =',F17.8,' Ry' &

    Parameters
    ----------
    pw_scf_out : str
        Name of the output file (typically ``'pw.out'`` or ``'scf.out'``).

    Returns
    -------
    eband : float
        The band energy.
    """
    energies, kpoints, f_occ, smearing_type, mu, kT = read_eps_nk_from_qe_pwo(
        pw_scf_out)
    nk, nbnd = energies.shape
    eband = np.zeros(energies.shape)

    if subset is None:
        for ik in range(nk):
            wk = kpoints[ik, 3] # weights
            for iband in range(nbnd):
                eband[ik, iband] = energies[ik, iband] * wk * f_occ[ik, iband]
    else:
        for ik in range(nk):
            wk = kpoints[ik, 3]
            for iband in subset:
                eband[ik, iband] = energies[ik, iband] * wk * f_occ[ik, iband]

    eband = eband.sum() / elphmod.misc.Ry

    return eband

def demet_from_qe_pwo(pw_scf_out, subset=None):
    """Calculate the entropy contribution :math:`-TS` to the total free energy.

    In Quantum ESPRESSO ``demet`` is calculated in ``gweights.f90``.

    Parameters
    ----------
    pw_scf_out : str
        The name of the output file (typically ``'pw.out'``).
    subset : list or array
        List of indices to pick only a subset of the bands for the integration.

    Returns
    -------
    demet : float
        The :math:`-TS` contribution to the total free energy.
    """
    energies, kpoints, f_occ, smearing_type, mu, kT = read_eps_nk_from_qe_pwo(
        pw_scf_out)
    nk, nbnd = energies.shape
    demet = np.zeros(energies.shape)

    f = elphmod.occupations.smearing(smearing_type)

    if subset is None:
        for ik in range(nk):
            wk = kpoints[ik, 3]
            for iband in range(nbnd):
                w1gauss = -f.entropy((energies[ik, iband] - mu) / kT)
                demet[ik, iband] = wk * kT * w1gauss

    else:
        for ik in range(nk):
            wk = kpoints[ik, 3]
            for iband in subset:
                w1gauss = -f.entropy((energies[ik, iband] - mu) / kT)
                demet[ik, iband] = wk * kT * w1gauss

    demet = demet.sum() / elphmod.misc.Ry

    return demet

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
        Arguments for :func:`elphmod.bravais.primitives`: Choose the right
        Bravais lattice (``ibrav``) and lattice constants (``a, b, c, ...``).

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
    bravais_vectors = elphmod.bravais.primitives(**kwargs)
    el = Model(seedname, divide_ndegen=False)

    R = np.empty(len(el.R))
    H = np.empty(len(el.R))

    # loop over all Wigner-seitz grid points
    for ii in range(len(el.R)):

        distance = np.empty((3, 3))
        for xi in range(3):
            distance[xi, :] = el.R[ii][xi] * bravais_vectors[xi, :]
        distance = distance.sum(axis=0)

        R[ii] = np.linalg.norm(distance)
        H[ii] = np.max(abs(el.data[ii])) / elphmod.misc.Ry

    return R, H

def read_energy_contributions_scf_out(filename):
    """Read energy contributions to the total energy ``scf`` output file.

    Parameters
    ----------
    filename : str
        Name of Quantum ESPRESSO's ``scf`` output file.

    Returns
    -------
    dict
        Energy contributions.
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

def read_pp_density(filename):
    r"""Read file *filout* with charge density generated by ``pp.x``.

    Calculate total charge

    .. math::

        Q = \int \rho(\vec r) dx dy dz =
            = \frac \Omega {n_1 n_2 n_3} \sum_{i j k} \rho_{i j k},

    where :math:`\Omega` is the unit-cell volume, :math:`n_r` are the FFT grid
    dimensions, and :math:`i, j, k` run over FFT real-space grid points.

    Parameters
    ----------
    filename : str
        Filename.

    Returns
    -------
    rho : ndarray
        Electronic charge density :math:`\rho_{i j k}` on FFT real-space grid
        points.
    tot_charge: float
        Total charge calculated from charge density :math:`\rho_{i j k}`.
    """
    with open(filename) as data:
        # read all words of current line:

        def cells():
            for line in data:
                words = line.split()

                if words:
                    return words

        # read table:

        def table(rows):
            return np.array([list(map(float, cells())) for row in range(rows)])

        # read FFT dimensions, ntyp, nat
        tmp = cells()
#        FFT_dim = list(map(int, tmp[:3]))
        FFT_dim = list(map(int, tmp[:6]))
        nat, ntyp = list(map(int, tmp[6:]))

        # read ibrav, celldm
        tmp = cells()
        ibrav = int(tmp[:1][0])
        celldm = list(map(float, tmp[1:]))

        # read unknown line
        tmp = cells()
        kin_energy_cutoff = float(tmp[2])

        # calculate unit cell and brillouin zone volume
        A = elphmod.bravais.primitives(ibrav)
        A[2, 2] = celldm[2]
        uc_volume = np.linalg.det(A) * celldm[0] ** 3
#        B1, B2, B3 = elphmod.bravais.reciprocals(*A)
#        B = np.stack((B1, B2, B3), axis=0)
#        print(B)
#        print(np.linalg.det(B) * (2 * np.pi / celldm[0]) ** 3)

        # read valence table
        at = []
        valence = []
        for line in range(ntyp):
            tmp = cells()
            at.append(tmp[1])
            valence.append(float(tmp[2]))

        # read tau table
        tau = table(nat)[0:4, 1:4]

        # read rho table
        # total number of real-space grid points:
        nr_points = np.prod(FFT_dim[:3])
        # separated into columns of 5:
        if divmod(nr_points, 5)[1] != 0:
            print('Warning: Total number of grid points is not divisible by 5!')
        rho = table(divmod(nr_points, 5)[0])

        tot_charge = rho.sum() / nr_points * uc_volume

        rho = rho.reshape(np.prod(FFT_dim[:3]))

        return rho, tot_charge

def read_rhoG_density(filename, ibrav, a=1.0, b=1.0, c=1.0):
    r"""Read the charge density output from Quantum ESPRESSO.

    The purpose of :math:`\rho(\vec G)` is to calculate the charge density in
    real space :math:`\rho(\vec r)` or the Hartree energy ``ehart``.

    Parameters
    ----------
    filename : str
        Filename.
    ibrav : integer
        Bravais lattice index (see ``pw.x`` input description).
    a, b, c : float
        Bravais lattice parameters.

    Returns
    -------
    rho_g : ndarray
        Electronic charge density :math:`\rho(\vec G)` on reciprocal-space grid
        points.
    g_vect : ndarray
        Reciprocal lattice vectors :math:`\vec G`.
    ngm_g : integer
        Number of reciprocal lattice vectors.
    uc_volume : float
        Unit-cell volume.
    mill_g : ndarray
        Miller indices.
    """
    with open(filename, 'rb') as f:
        # Moves the cursor 4 bytes to the right
        f.seek(4)

        gamma_only = bool(np.fromfile(f, dtype=np.int32, count=1)[0])
        ngm_g = np.fromfile(f, dtype=np.int32, count=1)[0]
        ispin = np.fromfile(f, dtype=np.int32, count=1)[0]

        # Move the cursor 8 byte to the right
        f.seek(8, 1)

        b1 = np.fromfile(f, dtype=np.float64, count=3)
        b2 = np.fromfile(f, dtype=np.float64, count=3)
        b3 = np.fromfile(f, dtype=np.float64, count=3)

        # Move the cursor 8 byte to the right
        f.seek(8, 1)

        mill_g = np.fromfile(f, dtype=np.int32, count=3 * ngm_g)
        mill_g = mill_g.reshape((ngm_g, 3))

        # Move the cursor 8 byte to the right
        f.seek(8, 1)

        rho_g = np.fromfile(f, dtype=np.complex128, count=ngm_g)

        # get primitive lattice vectors
        # (must be the same as scf output)
        A = elphmod.bravais.primitives(ibrav, a=a, b=b, c=c)
        A /= a

        B1, B2, B3 = elphmod.bravais.reciprocals(*A)

        # calculate unit cell volume (for Hartree energy)
        uc_volume = np.linalg.det(A) * (a / elphmod.misc.a0) ** 3

        # calculate reciprocal G vectors
        g_vect = np.empty((ngm_g, 3))
        for ii in range(ngm_g):
            g_vect[ii] = (mill_g[ii, 0] * B1 + mill_g[ii, 1] * B2
                + mill_g[ii, 2] * B3)

        return rho_g, g_vect, ngm_g, uc_volume, mill_g

def read_wfc(filename, ibrav, a=1.0, b=1.0, c=1.0):
    r"""Read the wave function output from Quantum ESPRESSO.

    .. math::

        \psi_{n \vec k}(\vec r) = \frac 1 {\sqrt V} \sum_{\vec G}
            c_{n, \vec k + \vec G} \E^{\I (\vec k + \vec G) \vec r}

    Parameters
    ----------
    filename : str
        Filename.
    ibrav : integer
        Bravais lattice index (see ``pw.x`` input description).
    a, b, c : float
        Bravais lattice parameters.

    Returns
    -------
    evc : ndarray
        Electronic wave function :math:`c_{n, \vec k + \vec G}`.
    igwx : integer
        Number of reciprocal lattice vectors.
    xk : ndarray
        k point.
    k_plus_G : ndarray
        k point plus reciprocal lattice vectors :math:`\vec G`.
    g_vect : ndarray
        Reciprocal lattice vectors :math:`\vec G`.
    mill_g : ndarray
        Miller indices.
    """
    with open(filename, 'rb') as f:
        # Moves the cursor 4 bytes to the right
        f.seek(4)

        ik = np.fromfile(f, dtype=np.int32, count=1)[0]
        xk = np.fromfile(f, dtype=np.float64, count=3)
        ispin = np.fromfile(f, dtype=np.int32, count=1)[0]
        gamma_only = bool(np.fromfile(f, dtype=np.int32, count=1)[0])
        scalef = np.fromfile(f, dtype=np.float64, count=1)[0]

        # Move the cursor 8 byte to the right
        f.seek(8, 1)

        ngw = np.fromfile(f, dtype=np.int32, count=1)[0]
        igwx = np.fromfile(f, dtype=np.int32, count=1)[0]
        npol = np.fromfile(f, dtype=np.int32, count=1)[0]
        nbnd = np.fromfile(f, dtype=np.int32, count=1)[0]

        # Move the cursor 8 byte to the right
        f.seek(8, 1)

        b1 = np.fromfile(f, dtype=np.float64, count=3)
        b2 = np.fromfile(f, dtype=np.float64, count=3)
        b3 = np.fromfile(f, dtype=np.float64, count=3)

        f.seek(8, 1)

        mill_g = np.fromfile(f, dtype=np.int32, count=3 * igwx)
        mill_g = mill_g.reshape((igwx, 3))

        evc = np.zeros((nbnd, npol * igwx), dtype=complex)

        f.seek(8, 1)
        for i in range(nbnd):
            evc[i, :] = np.fromfile(f, dtype=np.complex128, count=npol * igwx)
            f.seek(8, 1)

        # delta_mn = \sum_G \psi(m, G) * \psi(n, G)
        # print((evc[1, :].conj() * evc[0, :]).sum().real)

        # get primitive lattice vectors
        # (must be the same as scf output)
        A = elphmod.bravais.primitives(ibrav, a=a, b=b, c=c)
        A /= a

        # reciprocal lattice vectors for G
        B1, B2, B3 = elphmod.bravais.reciprocals(*A)

        # transform k point
        alat = a / elphmod.misc.a0
        xk = xk * alat / (2 * np.pi)

#        # normalize wfcs
#        uc_volume = np.linalg.det(A) * alat ** 3
#        evc /= np.sqrt(uc_volume)

        # calculate reciprocal G vectors
        g_vect = np.empty((igwx, 3))
        k_plus_G = np.empty((igwx, 3))

        for ii in range(igwx):
            g_vect[ii] = (mill_g[ii, 0] * B1 + mill_g[ii, 1] * B2
                + mill_g[ii, 2] * B3)
#            g_vect[ii] = (mill_g[ii, 0] * b1 + mill_g[ii, 1] * b2
#                + mill_g[ii, 2] * b3)

            k_plus_G[ii] = xk + g_vect[ii]

    return evc, igwx, xk, k_plus_G, g_vect, mill_g
