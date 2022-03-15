#!/usr/bin/env python3

# Copyright (C) 2017-2022 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import numpy as np

from . import bravais, misc, MPI
comm = MPI.comm

class Model(object):
    """Mass-spring model for the phonons.

    Parameters
    ----------
    flfrc : str
        File with interatomic force constants from ``q2r.x``.
    quadrupole_fmt : str
        File with quadrupole tensors in format suitable for ``epw.x``.
    apply_asr : bool
        Apply acoustic sum rule correction to force constants?
    apply_asr_simple : bool
        Apply simple acoustic sum rule correction to force constants? This sets
        the self force constant to minus the sum of all other force constants.
    apply_zasr : bool
        Apply acoustic sum rule correction to Born effective charges?
    apply_rsr : bool
        Apply rotation sum rule correction to force constants?
    lr : bool
        Compute long-range terms in case of polar material?
    lr2d : bool
        Compute long-range terms for two-dimensional system if `lr`?
    phid : ndarray
        Force constants if `flfrc` is omitted.
    amass : ndarray
        Atomic masses if `flfrc` is omitted.
    at : ndarray
        Bravais lattice vectors if `flfrc` is omitted.
    tau : ndarray
        Positions of basis atoms if `flfrc` is omitted.
    atom_order : list of str
        Ordered list of atoms if `flfrc` is omitted.
    epsil : ndarray
        Dielectric tensor if `flfrc` is omitted.
    zeu : ndarray
        Born effective charges if `flfrc` is omitted.
    Q : ndarray
        Quadrupole tensors if `quadrupole_fmt` is omitted.
    divide_mass : bool
        Divide force constants and Born effective charges by atomic masses?
    divide_ndegen : bool
        Divide force constants by degeneracy of Wigner-Seitz point? Only
        ``True`` yields correct phonons. ``False`` should only be used for
        debugging.

    Attributes
    ----------
    M : ndarray
        Atomic masses.
    a : ndarray
        Bravais lattice vectors.
    r : ndarray
        Positions of basis atoms.
    R : ndarray
        Lattice vectors of Wigner-Seitz supercell.
    l : ndarray.
        Bond lengths.
    atom_order : list of str
        Ordered list of atoms.
    eps : ndarray
        Dielectric tensor.
    Z : ndarray
        Born effective charges divided by square root of atomic masses.
    Q : ndarray
        Quadrupole tensors divided by square root of atomic masses.
    data : ndarray
        Interatomic force constants divided by atomic masses.
    divide_mass : bool
        Have force constants and Born charges been divided by atomic masses?
    size : int
        Number of displacement directions/bands.
    nat : int
        Number of atoms.
    cells : list of tuple of int, optional
        Lattice vectors of unit cells if the model describes a supercell.
    lr : bool
        Compute long-range terms?
    lr2d : bool
        Compute long-range terms for two-dimensional system if `lr`?
    b : ndarray
        Reciprocal primitive vectors if `lr`.
    prefactor : float
        Prefactor of long-range terms if `lr`.
    r_eff : float
        Effective thickness if `lr2d`.
    scale : float
        Relevant scaling factor if `lr`.
    D0_lr : ndarray
        Constant part of long-range correction to dynamical matrix if `lr`.
    """
    def D(self, q1=0, q2=0, q3=0):
        """Set up dynamical matrix for arbitrary q point."""

        q = np.array([q1, q2, q3])

        # Sign convention in do_q3r.f90 of QE:
        # 231  CALL cfft3d ( phid (:,j1,j2,na1,na2), &
        # 232       nr1,nr2,nr3, nr1,nr2,nr3, 1, 1 )
        # 233  phid(:,j1,j2,na1,na2) = &
        # 234       phid(:,j1,j2,na1,na2) / DBLE(nr1*nr2*nr3)
        # The last argument of cfft3d is the sign (+1).

        Dq = np.einsum('Rxy,R->xy', self.data, np.exp(-1j * self.R.dot(q)))

        if self.lr:
            Dq += self.D_lr(q1, q2, q3)

        return Dq

    def D_lr(self, q1=0, q2=0, q3=0):
        """Calculate long-range part of dynamical matrix."""

        Dq = self.D0_lr.copy()

        for factor, d, q in self.generate_long_range(q1, q2, q3):
            Dq += factor * np.outer(d, d.conj())

            if self.Q is not None:
                Dq += factor * (np.outer(d, q.conj()) + np.outer(q, d.conj())
                    + np.outer(q, q.conj()))

        return Dq

    def C(self, R1=0, R2=0, R3=0):
        """Get interatomic force constants for arbitrary lattice vector."""

        index = misc.vector_index(self.R, (R1, R2, R3))

        if index is None:
            return np.zeros_like(self.data[0])
        else:
            return self.data[index]

    def __init__(self, flfrc=None, quadrupole_fmt=None, apply_asr=False,
        apply_asr_simple=False, apply_zasr=False, apply_rsr=False, lr=True,
        lr2d=False, phid=np.zeros((1, 1, 1, 1, 1, 3, 3)), amass=np.ones(1),
        at=np.eye(3), tau=np.zeros((1, 3)), atom_order=['X'], epsil=None,
        zeu=None, Q=None, divide_mass=True, divide_ndegen=True):

        if comm.rank == 0:
            if flfrc is None:
                model = phid.copy(), amass, at, tau, atom_order, epsil, zeu
            else:
                model = read_flfrc(flfrc)

            if quadrupole_fmt is not None:
                Q = read_quadrupole_fmt(quadrupole_fmt)

            # optionally, apply acoustic sum rule:

            if apply_asr_simple:
                asr(model[0])

            if apply_zasr:
                zasr(model[-1])
        else:
            model = None

        model = comm.bcast(model)

        self.Q = comm.bcast(Q)

        self.M, self.a, self.r, self.atom_order, self.eps, self.Z = model[1:]
        self.R, self.data, self.l = short_range_model(*model[:-3],
            divide_mass=divide_mass, divide_ndegen=divide_ndegen)
        self.size = self.data.shape[1]
        self.nat = self.size // 3
        self.divide_mass = divide_mass

        if apply_asr or apply_rsr:
            sum_rule_correction(self, asr=apply_asr, rsr=apply_rsr,
                divide_mass=divide_mass)

        self.lr = lr and self.eps is not None and self.Z is not None

        if self.lr:
            self.lr2d = lr2d
            self.prepare_long_range()

    def prepare_long_range(self, alpha=1.0, G_max=14.0):
        """Prepare calculation of long-range terms for polar materials.

        The following two routines are based on ``rgd_blk`` and ``rgd_blk_epw``
        of Quantum ESPRESSO and the EPW code.

        Copyright (C) 2010-2016 S. Ponce', R. Margine, C. Verdi, F. Giustino
        Copyright (C) 2001-2012 Quantum ESPRESSO group

        Please refer to:

        * Phonons: Gonze et al., Phys. Rev. B 50, 13035 (1994)
        * Coupling: Verdi and Giustino, Phys. Rev. Lett. 115, 176401 (2015)
        * 2D case: Sohier, Calandra, and Mauri, Phys. Rev. B 94, 085415 (2016)
        * Quadrupoles: Ponce' et al., Phys. Rev. Research 3, 043022 (2021)

        Parameters
        ----------
        alpha : float
            Ewald parameter.
        G_max : float
            Cutoff for reciprocal lattice vectors.
        """
        self.b = np.array(bravais.reciprocals(*self.a))

        e2 = 2.0 # square of electron charge in Rydberg units

        if self.lr2d:
            c = 1 / self.b[2, 2]
            self.r_eff = (self.eps[:2, :2] - np.eye(2)) * c / 2

            area = np.linalg.norm(np.cross(self.a[0], self.a[1]))
            self.prefactor = 2 * np.pi * e2 / area
        else:
            volume = abs(np.dot(self.a[0], np.cross(self.a[1], self.a[2])))
            self.prefactor = 4 * np.pi * e2 / volume

        a = np.linalg.norm(self.a[0])
        self.scale = 4 * alpha * (2 * np.pi / a) ** 2

        nr = 1 + (np.sqrt(self.scale * G_max)
            / np.linalg.norm(2 * np.pi * self.b, axis=1)).astype(int)

        if self.lr2d:
            nr[2] = 0

        self.G = []

        for m1 in range(-nr[0], nr[0] + 1):
            for m2 in range(-nr[1], nr[1] + 1):
                for m3 in range(-nr[2], nr[2] + 1):
                    G = 2 * np.pi * (m1 * self.b[0] + m2 * self.b[1]
                        + m3 * self.b[2])

                    if self.lr2d:
                        GeG = (G ** 2).sum()
                    else:
                        GeG = np.einsum('i,ij,j', G, self.eps, G)

                    if GeG < self.scale * G_max:
                        self.G.append(G)

        self.D0_lr = np.zeros((self.size, self.size), dtype=complex)
        Dq0_lr = self.D_lr()

        for na1 in range(self.nat):
            for na2 in range(self.nat):
                self.D0_lr[3 * na1:3 * na1 + 3, 3 * na1:3 * na1 + 3] -= (
                    Dq0_lr[3 * na1:3 * na1 + 3, 3 * na2:3 * na2 + 3])

        if self.divide_mass:
            for na in range(self.nat):
                self.D0_lr[3 * na:3 * na + 3, :] /= np.sqrt(self.M[na])
                self.D0_lr[:, 3 * na:3 * na + 3] /= np.sqrt(self.M[na])

                self.Z[na] /= np.sqrt(self.M[na])

                if self.Q is not None:
                    self.Q[na] /= np.sqrt(self.M[na])

    def generate_long_range(self, q1=0, q2=0, q3=0, eps=1e-8):
        r"""Generate long-range terms.

        Parameters
        ----------
        q : ndarray
            q point in reciprocal lattice units :math:`q_i \in [0, 2 \pi)`.
        eps : float
            Tolerance for vanishing lattice vectors.

        Yields
        ------
        float
            Relevant factor.
        ndarray
            Dipole term.
        ndarray
            Quadrupole term.
        """
        for G in self.G:
            K = G + q1 * self.b[0] + q2 * self.b[1] + q3 * self.b[2]

            if self.lr2d:
                K2K = (K[:2] ** 2).sum()

                if K2K < eps:
                    KrK = 0.0
                else:
                    KrK = np.einsum('i,ij,j', K[:2], self.r_eff, K[:2]) / K2K

                KeK = (K ** 2).sum()
            else:
                KeK = np.einsum('i,ij,j', K, self.eps, K)

            if KeK > eps:
                factor = self.prefactor * np.exp(-KeK / self.scale)
                factor /= np.sqrt(KeK) + KrK * KeK if self.lr2d else KeK

                exp = np.exp(1j * np.dot(self.r, K))
                exp = exp[:, np.newaxis]

                d = np.dot(K, self.Z) * exp
                d = d.ravel()

                if self.Q is not None:
                    q = 0.5j * K.dot(self.Q).dot(K) * exp
                    q = q.ravel()
                else:
                    q = None

                yield factor, d, q

    def symmetrize(self):
        r"""Symmetrize dynamical matrix.

        .. math::

            D_{\vec q} = D_{\vec q}^\dagger,
            D_{\vec R} = D_{-\vec R}^\dagger
        """
        if comm.rank == 0:
            status = misc.StatusBar(len(self.R),
                title='symmetrize dynamical matrix')

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
        """Map mass-spring model onto supercell.

        Parameters
        ----------
        N1, N2, N3 : tuple of int or int, default 1
            Supercell lattice vectors in units of primitive lattice vectors.
            Multiples of single primitive vector can be defined via a scalar
            integer, linear combinations via a 3-tuple of integers.

        Returns
        -------
        object
            Mass-spring model for supercell.
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

        ph = Model()
        ph.M = np.tile(self.M, N)
        ph.a = np.dot(np.array([N1, N2, N3]), self.a)
        ph.atom_order = list(self.atom_order) * N
        ph.size = self.size * N
        ph.nat = self.nat * N
        ph.N = [tuple(N1), tuple(N2), tuple(N3)]

        ph.cells = []

        if comm.rank == 0:
            for n1 in range(N):
                for n2 in range(N):
                    for n3 in range(N):
                        indices = n1 * N1 + n2 * N2 + n3 * N3

                        if np.all(indices % N == 0):
                            ph.cells.append(tuple(indices // N))

            assert len(ph.cells) == N

        ph.cells = comm.bcast(ph.cells)

        ph.r = np.array([
            n1 * self.a[0] + n2 * self.a[1] + n3 * self.a[2] + self.r[na]
            for n1, n2, n3 in ph.cells
            for na in range(self.nat)])

        if comm.rank == 0:
            const = dict()

            status = misc.StatusBar(len(self.R),
                title='map force constants onto supercell')

            for n in range(len(self.R)):
                for i, cell in enumerate(ph.cells):
                    R = self.R[n] + np.array(cell)

                    R1, r1 = divmod(np.dot(R, B1), N)
                    R2, r2 = divmod(np.dot(R, B2), N)
                    R3, r3 = divmod(np.dot(R, B3), N)

                    R = R1, R2, R3

                    indices = r1 * N1 + r2 * N2 + r3 * N3
                    j = ph.cells.index(tuple(indices // N))

                    A = i * self.size
                    B = j * self.size

                    if R not in const:
                        const[R] = np.zeros((ph.size, ph.size))

                    const[R][B:B + self.size, A:A + self.size] = self.data[n]

                status.update()

            ph.R = np.array(list(const.keys()), dtype=int)
            ph.data = np.array(list(const.values()))

            count = len(const)
            const.clear()
        else:
            count = None

        count = comm.bcast(count)

        if comm.rank != 0:
            ph.R = np.empty((count, 3), dtype=int)
            ph.data = np.empty((count, ph.size, ph.size))

        comm.Bcast(ph.R)
        comm.Bcast(ph.data)

        return ph

    def unit_cell(self):
        """Map mass-spring model back to unit cell.

        See Also
        --------
        supercell
        """
        ph = Model()
        ph.size = self.size // len(self.cells)
        ph.nat = self.nat // len(self.cells)
        ph.M = self.M[:ph.nat]
        ph.atom_order = self.atom_order[:ph.nat]
        ph.r = self.r[:ph.nat]

        B1 = np.cross(self.N[1], self.N[2])
        B2 = np.cross(self.N[2], self.N[0])
        B3 = np.cross(self.N[0], self.N[1])
        ph.a = np.dot(np.array([B1, B2, B3]).T, self.a) / len(self.cells)

        if comm.rank == 0:
            const = dict()

            status = misc.StatusBar(len(self.R),
                title='map force constants back to unit cell')

            for n in range(len(self.R)):
                for i, cell in enumerate(self.cells):
                    C = self.data[n, i * ph.size:(i + 1) * ph.size, :ph.size]

                    if np.any(C != 0):
                        R = tuple(np.dot(self.R[n], self.N) + np.array(cell))
                        const[R] = C

                status.update()

            ph.R = np.array(list(const.keys()), dtype=int)
            ph.data = np.array(list(const.values()))

            count = len(const)
            const.clear()
        else:
            count = None

        count = comm.bcast(count)

        if comm.rank != 0:
            ph.R = np.empty((count, 3), dtype=int)
            ph.data = np.empty((count, ph.size, ph.size))

        comm.Bcast(ph.R)
        comm.Bcast(ph.data)

        return ph

    def order_atoms(self, *order):
        """Reorder atoms.

        Together with :func:`shift_atoms`, this function helps reconcile
        inconsistent definitions of the basis/motif of the Bravais lattice.

        Parameters
        ----------
        *order : int
            New atom order.
        """
        self.data = self.data.reshape((len(self.R), self.nat, 3, self.nat, 3))

        self.data = self.data[:, order, :, :, :]
        self.data = self.data[:, :, :, order, :]

        self.data = self.data.reshape((len(self.R), self.size, self.size))

        self.M = [self.M[na] for na in order]
        self.atom_order = [self.atom_order[na] for na in order]
        self.r = [self.r[na] for na in order]

    def shift_atoms(self, s, S):
        """Move selected atoms across unit-cell boundary.

        Together with :func:`order_atoms`, this function helps reconcile
        inconsistent definitions of the basis/motif of the bravais lattice.

        Parameters
        ----------
        s : slice
            Slice of atom indices corresponding to selected basis atom(s).
        S : tuple of int
            Shift of as multiple of primitive lattice vectors.
        """
        self.data = self.data.reshape((len(self.R), self.nat, 3, self.nat, 3))

        S = np.asarray(S)
        data = self.data.copy()

        old_R = set(range(len(self.R)))
        new_R = []
        new_C = []

        for i in range(len(self.R)):
            R = self.R[i] + S
            j = misc.vector_index(self.R, R)

            if j is None:
                C = np.zeros((self.nat, 3, self.nat, 3))
                C[s, :, :, :] = self.data[i, s, :, :, :]
                C[s, :, s, :] = 0.0
                new_R.append(R)
                new_C.append(C)
            else:
                old_R.remove(j)
                data[j, s, :, :, :] = self.data[i, s, :, :, :]
                data[j, s, :, s, :] = self.data[j, s, :, s, :]

        for j in old_R:
            data[j, s, :, :, :] = 0.0
            data[j, s, :, s, :] = self.data[j, s, :, s, :]

        self.R = np.concatenate((self.R, new_R))
        self.data = np.concatenate((data, new_C))

        self.data = self.data.reshape((len(self.R), self.size, self.size))

        self.standardize()

        self.r[s] += np.dot(S, self.a)

    def standardize(self):
        """Standardize mass-spring data.

        - Keep only nonzero force-constant matrices.
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
            self.data = np.empty((count, self.size, self.size))

        comm.Bcast(self.R)
        comm.Bcast(self.data)

    def decay(self):
        """Plot force constants as a function of bond length.

        Returns
        -------
        ndarray
            Bond lengths.
        ndarray
            Frobenius norm of force-constant matrices.
        """
        l = self.l.flatten()
        C = np.linalg.norm(self.data.reshape((len(self.R),
            self.nat, 3, self.nat, 3)), ord='fro', axis=(2, 4)).flatten()

        nonzero = np.where(C > 0)

        return l[nonzero], C[nonzero]

def group(n, size=3):
    """Create slice of dynamical matrix belonging to `n`-th atom."""

    return slice(n * size, (n + 1) * size)

def read_fildyn(fildyn, divide_mass=True):
    """Read file *fildyn* as created by Quantum ESPRESSO's ``ph.x``."""

    header = [] # header information as lines of plain text

    qpoints = [] # (equivalent) q points as lines of plain text
    dynmats = [] # corresponding dynamical matrices as complex NumPy arrays

    headline = 'Dynamical  Matrix in cartesian axes'

    with open(fildyn) as data:
        def headnext():
            line = next(data)
            header.append(line)
            return line

        headnext()
        headnext()

        ntyp, nat = map(int, headnext().split()[:2])

        amass = [float(headnext().split()[-1]) for _ in range(ntyp)]
        amass = [amass[int(headnext().split()[1]) - 1] for _ in range(nat)]

        headnext()

        while True:
            line = next(data)
            if not headline in line:
                footer = line
                break

            next(data)
            qpoints.append(np.array(list(map(float, next(data).split()[3:6]))))
            next(data)

            dim = 3 * nat
            dynmats.append(np.empty((dim, dim), dtype=complex))

            for i in range(nat):
                for j in range(nat):
                    next(data)
                    for n in range(3):
                        cols = list(map(float, next(data).split()))
                        for m in range(3):
                            dynmats[-1][group(i), group(j)][n, m] = complex(
                                *cols[group(m, 2)])

            next(data)

        for line in data:
            footer += line

    if divide_mass:
        for p in range(len(dynmats)):
            for i in range(nat):
                dynmats[p][group(i), :] /= np.sqrt(amass[i])
                dynmats[p][:, group(i)] /= np.sqrt(amass[i])

    return ''.join(header), qpoints, dynmats, footer, amass

def write_fildyn(fildyn, header, qpoints, dynmats, footer, amass,
        divide_mass=True):
    """Write file *fildyn* as created by Quantum ESPRESSO's ``ph.x``."""

    nat = len(amass)

    if divide_mass:
        for p in range(len(dynmats)):
            for i in range(nat):
                dynmats[p][group(i), :] *= np.sqrt(amass[i])
                dynmats[p][:, group(i)] *= np.sqrt(amass[i])

    headline = 'Dynamical  Matrix in cartesian axes'

    with open(fildyn, 'w') as data:
        data.write(header)

        for p in range(len(dynmats)):
            data.write('     %s\n\n' % headline)
            data.write('     q = ( ')

            for coordinate in qpoints[p]:
                data.write('%14.9f' % coordinate)

            data.write(' ) \n\n')

            for i in range(nat):
                for j in range(nat):
                    data.write('%5d%5d\n' % (i + 1, j + 1))
                    for n in range(3):
                        data.write('  '.join('%12.8f%12.8f' % (z.real, z.imag)
                            for z in dynmats[p][group(i), group(j)][n]))
                        data.write('\n')

            data.write('\n')

        data.write(footer)

def read_q(fildyn0):
    """Read list of irreducible q points from *fildyn0*."""

    with open(fildyn0) as data:
        return [list(map(float, line.split()[:2]))
            for line in data if '.' in line]

def write_q(fildyn0, q, nq):
    """Write list of irreducible q points to *fildyn0*."""

    with open(fildyn0, 'w') as data:
        data.write('%4d%4d%4d\n' % (nq, nq, 1))
        data.write('%4d\n' % len(q))

        for qxy in q:
            data.write('%19.15f%19.15f%19.15f\n' % (qxy[0], qxy[1], 0.0))

def fildyn_freq(fildyn='matdyn'):
    """Create *fildyn.freq* as created by Quantum ESPRESSO's ``ph.x``

    Parameters
    ----------
    fildyn : str, default 'matdyn'
        Prefix of files with dynamical matrices.
    """
    if comm.rank != 0:
        return

    nq = len(read_q('%s0' % fildyn))

    with open('%s.freq' % fildyn, 'w') as freq:
        for iq in range(nq):
            header, qpoints, dynmats, footer, amass = read_fildyn('%s%d'
                % (fildyn, iq + 1))

            w = sgnsqrt(np.linalg.eigvalsh(dynmats[0])) * misc.Ry / misc.cmm1

            if iq == 0:
                freq.write(' &plot nbnd=%4d, nks=%4d /\n' % (len(w), nq))

            freq.write('%20.6f %9.6f %9.6f\n' % tuple(qpoints[0]))

            for nu, wnu in enumerate(w, 1):
                freq.write('%10.4f' % wnu)

                if not nu % 6 or nu == len(w):
                    freq.write('\n')

def read_flfrc(flfrc):
    """Read file *flfrc* with force constants generated by ``q2r.x``."""

    with open(flfrc) as data:
        # read all words of current line:

        def cells():
            return data.readline().split()

        # read table:

        def table(rows):
            return np.array([list(map(float, cells())) for row in range(rows)])

        # read crystal structure:

        tmp = cells()
        ntyp, nat, ibrav = list(map(int, tmp[:3]))
        celldm = list(map(float, tmp[3:]))

        # see Modules/latgen.f90 of Quantum ESPRESSO:

        if ibrav:
            at = bravais.primitives(ibrav, celldm=celldm, bohr=True)
        else: # free
            at = table(3) * celldm[0]

        # read palette of atomic species and masses:

        atm = []
        amass = np.empty(ntyp)

        for nt in range(ntyp):
            tmp = cells()

            atm.append(tmp[1][1:3])
            amass[nt] = float(tmp[-1])

        # read types and positions of individual atoms:

        ityp = np.empty(nat, dtype=int)
        tau = np.empty((nat, 3))

        for na in range(nat):
            tmp = cells()

            ityp[na] = int(tmp[1]) - 1
            tau[na, :] = list(map(float, tmp[2:5]))

        tau *= celldm[0]

        atom_order = []

        for index in ityp:
            atom_order.append(atm[index].strip())

        # read macroscopic dielectric function and effective charges:

        lrigid = cells()[0] == 'T'

        if lrigid:
            epsil = table(3)

            zeu = np.empty((nat, 3, 3))

            for na in range(nat):
                na = int(cells()[0]) - 1
                zeu[na] = table(3)
        else:
            epsil = zeu = None

        # read interatomic force constants:

        nr1, nr2, nr3 = map(int, cells())

        phid = np.empty((nat, nat, nr1, nr2, nr3, 3, 3))

        for j1 in range(3):
            for j2 in range(3):
                for na1 in range(nat):
                    for na2 in range(nat):
                        cells() # skip line with j1, j2, na2, na2

                        for m3 in range(nr3):
                            for m2 in range(nr2):
                                for m1 in range(nr1):
                                    phid[na1, na2, m1, m2, m3, j1, j2] \
                                        = float(cells()[-1])

    # return force constants, masses, and geometry:

    return [phid, amass[ityp], at, tau, atom_order, epsil, zeu]

def read_quadrupole_fmt(quadrupole_fmt):
    """Read file *quadrupole.fmt* suitable for ``epw.x``."""

    Q = []

    with open(quadrupole_fmt) as data:
        next(data)

        for line in data:
            cols = line.split()

            na, i = [int(col) - 1 for col in cols[:2]]

            Qxx, Qyy, Qzz, Qyz, Qxz, Qxy = list(map(float, cols[2:]))

            while len(Q) <= na:
                Q.append(np.zeros((3, 3, 3)))

            Q[na][i, 0, 0] = Qxx
            Q[na][i, 1, 1] = Qyy
            Q[na][i, 2, 2] = Qzz
            Q[na][i, 1, 2] = Qyz
            Q[na][i, 2, 1] = Qyz
            Q[na][i, 0, 2] = Qxz
            Q[na][i, 2, 0] = Qxz
            Q[na][i, 0, 1] = Qxy
            Q[na][i, 1, 0] = Qxy

    return np.array(Q)

def asr(phid):
    """Apply simple acoustic sum rule correction to force constants."""

    nat, nr1, nr2, nr3 = phid.shape[1:5]

    for na1 in range(nat):
        phid[na1, na1, 0, 0, 0] = -sum(
        phid[na2, na1, m1, m2, m3]
            for na2 in range(nat)
            for m1 in range(nr1)
            for m2 in range(nr2)
            for m3 in range(nr3)
            if na1 != na2 or m1 or m2 or m3)

def zasr(Z):
    """Apply acoustic sum rule correction to Born effective charges."""

    Z -= np.average(Z, axis=0)

def sum_rule_correction(ph, asr=True, rsr=True, eps=1e-15, report=True,
        divide_mass=True):
    """Apply sum rule correction to force constants.

    Parameters
    ----------
    ph : object
        Mass-spring model for the phonons.
    asr : bool
        Enforce acoustic sum rule?
    rsr : bool
        Enforce Born-Huang rotation sum rule?
    eps : float
        Smallest safe absolute value of divisor.
    report : bool
        Print sums before and after correction?
    divide_mass : bool
        Have the force constants of `ph` been divided by atomic masses?
    """
    if comm.rank != 0:
        comm.Bcast(ph.data)
        return

    # define sums that should be zero:

    def acoustic_sum():
        zero = 0.0
        for l in range(ph.nat):
            for x in range(3):
                for y in range(3):
                    S = 0.0
                    for n in range(len(R)):
                        for k in range(ph.nat):
                            S += C[n, k, x, l, y]
                    zero += abs(S)
        return zero

    def rotation_sum():
        zero = 0.0
        for l in range(ph.nat):
            for x1 in range(3):
                for x2 in range(3):
                    for y in range(3):
                        S = 0.0
                        for n in range(len(R)):
                            for k in range(ph.nat):
                                S += (C[n, k, x1, l, y]
                                    * (R[n, x2] + ph.r[k, x2]))
                                S -= (C[n, k, x2, l, y]
                                    * (R[n, x1] + ph.r[k, x1]))
                        zero += abs(S)
        return zero

    # prepare lattice vectors and force constants:

    R = np.einsum('xy,nx->ny', ph.a, ph.R)
    C = ph.data.reshape((len(R), ph.nat, 3, ph.nat, 3))

    if divide_mass:
        for na in range(ph.nat):
            C[:, na, :, :, :] *= np.sqrt(ph.M[na])
            C[:, :, :, na, :] *= np.sqrt(ph.M[na])

    # check if sum rules are already fulfilled (before correction):

    if report:
        print('Acoustic sum (before): %g' % acoustic_sum())
        print('Rotation sum (before): %g' % rotation_sum())

    # determine list of constraints:

    c = []

    if asr:
        for l in range(ph.nat):
            for x in range(3):
                for y in range(3):
                    c.append(np.zeros_like(C))
                    for n in range(len(R)):
                        for k in range(ph.nat):
                            c[-1][n, k, x, l, y] += 1.0

    if rsr:
        for l in range(ph.nat):
            for x1 in range(3):
                for x2 in range(3):
                    for y in range(3):
                        c.append(np.zeros_like(C))
                        for n in range(len(R)):
                            for k in range(ph.nat):
                                c[-1][n, k, x1, l, y] += R[n, x2] + ph.r[k, x2]
                                c[-1][n, k, x2, l, y] -= R[n, x1] + ph.r[k, x1]

    # symmetrize constraints (the force constants must be symmetric):

    minusR = [np.argmax(np.all(ph.R == -ph.R[n], axis=1))
        for n in range(len(ph.R))]

    for i in range(len(c)):
        c[i] += c[i][minusR].transpose((0, 3, 4, 1, 2))

    c.append(C)

    # orthogonalize constraints and force constants via Gram-Schmidt method:

    cc = np.empty(len(c))
    for i in range(len(c)):
        for j in range(i):
            if cc[j] > eps:
                c[i] -= (c[j] * c[i]).sum() / cc[j] * c[j]
        cc[i] = (c[i] * c[i]).sum()

    # check if sum rules are really fulfilled now (after correction):

    if report:
        print('Acoustic sum (after): %g' % acoustic_sum())
        print('Rotation sum (after): %g' % rotation_sum())

    # redo division by atomic masses:

    if divide_mass:
        for na in range(ph.nat):
            C[:, na, :, :, :] /= np.sqrt(ph.M[na])
            C[:, :, :, na, :] /= np.sqrt(ph.M[na])

    comm.Bcast(ph.data)

def short_range_model(phid, amass, at, tau, eps=1e-7, divide_mass=True,
        divide_ndegen=True):
    """Map force constants onto Wigner-Seitz cell and divide by masses."""

    nat, nr1, nr2, nr3 = phid.shape[1:5]

    supercells = [-1, 0, 1] # indices of central and neighboring supercells

    C = np.empty((3, 3))

    const = dict()

    N = 0 # counter for parallelization

    for m1 in range(nr1):
        for m2 in range(nr2):
            for m3 in range(nr3):
                N += 1

                if N % comm.size != comm.rank:
                    continue

                # determine equivalent unit cells within considered supercells:

                copies = np.array([[
                        m1 + M1 * nr1,
                        m2 + M2 * nr2,
                        m3 + M3 * nr3,
                        ]
                    for M1 in supercells
                    for M2 in supercells
                    for M3 in supercells
                    ])

                # calculate corresponding translation vectors:

                shifts = [np.dot(copy, at) for copy in copies]

                for na1 in range(nat):
                    for na2 in range(nat):
                        # find equivalent bond(s) within Wigner-Seitz cell:

                        bonds = [r + tau[na1] - tau[na2] for r in shifts]
                        lengths = [np.sqrt(np.dot(r, r)) for r in bonds]
                        length = min(lengths)

                        selected = copies[np.where(abs(lengths - length) < eps)]

                        # undo supercell double counting and divide by masses:

                        C[...] = phid[na1, na2, m1, m2, m3]

                        if divide_ndegen:
                            C /= len(selected)

                        if divide_mass:
                            C /= np.sqrt(amass[na1] * amass[na2])

                        # save data for dynamical matrix calculation:

                        for R in selected:
                            R = tuple(R)

                            if R not in const:
                                const[R] = [
                                    np.zeros((3 * nat, 3 * nat)),
                                    np.zeros((nat, nat))]

                            const[R][0][3 * na1:3 * na1 + 3,
                                        3 * na2:3 * na2 + 3] = C

                            const[R][1][na1, na2] = length

    # convert dictionary into arrays:

    my_count = len(const)
    my_cells = np.array(list(const.keys()), dtype=np.int8)
    my_const = np.empty((my_count, 3 * nat, 3 * nat))
    my_bonds = np.empty((my_count, nat, nat))

    for i, (c, l) in enumerate(const.values()):
        my_const[i] = c
        my_bonds[i] = l

    # gather data of all processes:

    my_counts = np.array(comm.allgather(my_count))
    count = my_counts.sum()

    cells = np.empty((count, 3), dtype=np.int8)
    const = np.empty((count, 3 * nat, 3 * nat))
    bonds = np.empty((count, nat, nat))

    comm.Allgatherv(my_cells, (cells, my_counts * 3))
    comm.Allgatherv(my_const, (const, my_counts * (3 * nat) ** 2))
    comm.Allgatherv(my_bonds, (bonds, my_counts * nat ** 2))

    # (see cdef _p_message message_vector in mpi4py/src/mpi4py/MPI/msgbuffer.pxi
    # for possible formats of second argument 'recvbuf')

    return cells, const, bonds

def sgnsqrt(w2):
    """Calculate signed square root."""

    return np.sign(w2) * np.sqrt(np.absolute(w2))

def polarization(e, path, angle=60):
    """Characterize as in-plane longitudinal/transverse or out-of-plane."""

    bands = e.shape[2]

    mode = np.empty((len(path), bands, 3))

    nat = bands // 3

    x = slice(0, None, 3)
    y = slice(1, None, 3)
    z = slice(2, None, 3)

    a1, a2 = bravais.translations(180 - angle)
    b1, b2 = bravais.reciprocals(a1, a2)

    for n, q in enumerate(path):
        q = q[0] * b1 + q[1] * b2
        Q = np.sqrt(q.dot(q))

        centered = Q < 1e-10

        if centered:
            q = np.array([1, 0])
        else:
            q /= Q

        for band in range(bands):
            L = sum(abs(e[n, x, band] * q[0] + e[n, y, band] * q[1]) ** 2)
            T = sum(abs(e[n, x, band] * q[1] - e[n, y, band] * q[0]) ** 2)
            Z = sum(abs(e[n, z, band]) ** 2)

            if centered:
                L = T = (L + T) / 2

            mode[n, band, :] = [L, T, Z]

    return mode

def q2r(ph, D_irr=None, q_irr=None, nq=None, D_full=None, angle=60,
        apply_asr=False, apply_asr_simple=False, apply_rsr=False,
        divide_mass=True, divide_ndegen=True):
    """Interpolate dynamical matrices given for irreducible wedge of q points.

    This function replaces `interpolate_dynamical_matrices`, which depends on
    Quantum ESPRESSO. For 2D lattices, it is sufficient to provide dynamical
    matrices `D_irr` for the irreducible q points `q_irr`. Here, for the square
    lattice, the rotation symmetry (90 degrees) is currently disabled! In turn,
    for 1D and 2D lattices, dynamical matrices `D_full` on the complete uniform
    q-point mesh must be given.

    Parameters
    ----------
    ph : object
        Mass-spring model.
    D_irr : list of square arrays
        Dynamical matrices for all irreducible q points.
    q_irr : list of 2-tuples
        Irreducible q points in crystal coordinates with period :math:`2 \pi`.
    nq : int or tuple of int
        Number of q points per dimension, i.e., size of uniform mesh. Different
        numbers of q points along different axes can be specified via a tuple.
        Alternatively, `nq` is inferred from the shape of `D_full`.
    D_full : ndarray
        Dynamical matrices on complete uniform q-point mesh.
    angle : float
        Angle between mesh axes in degrees.
    apply_asr : bool
        Enforce acoustic sum rule by overwriting self force constants?
    apply_asr_simple : bool
        Apply simple acoustic sum rule correction to force constants? This sets
        the self force constant to minus the sum of all other force constants.
    apply_rsr : bool
        Enforce rotation sum rule by overwriting self force constants?
    divide_mass : bool
        Divide force constants by atomic masses?
    divide_ndegen : bool
        Divide force constants by degeneracy of Wigner-Seitz point? Only
        ``True`` yields correct phonons. ``False`` should only be used for
        debugging.
    """
    if D_full is None:
        D_full = np.empty((nq, nq, ph.size, ph.size), dtype=complex)

        def rotation(phi, n=1):
            block = np.array([
                [np.cos(phi), -np.sin(phi), 0],
                [np.sin(phi),  np.cos(phi), 0],
                [0,            0,           1],
                ])

            return np.kron(np.eye(n), block)

        def reflection(n=1):
            return np.diag([-1, 1, 1] * n)

        def apply(A, U):
            return np.einsum('ij,jk,kl->il', U, A, U.T.conj())

        a1, a2 = bravais.translations(180 - angle)
        b1, b2 = bravais.reciprocals(a1, a2)

        r = ph.r[:, :2].T / ph.a[0, 0]

        scale = nq / (2 * np.pi)

        # rotation currently disabled for square lattice:

        angles = [0] if angle == 90 else [0, 2 * np.pi / 3, 4 * np.pi / 3]

        for iq, (q1, q2) in enumerate(q_irr):
            q0 = q1 * b1 + q2 * b2

            for phi in angles:
                for reflect in False, True:
                    q = np.dot(rotation(phi)[:2, :2], q0)
                    D = apply(D_irr[iq], rotation(phi, ph.nat))

                    if reflect:
                        q = np.dot(reflection()[:2, :2], q)
                        D = apply(D, reflection(ph.nat))

                    # only valid if atom is mapped onto image of itself:

                    phase = np.exp(1j * np.array(np.dot(q0 - q, r)))

                    for n in range(ph.nat):
                        D[3 * n:3 * n + 3, :] *= phase[n].conj()
                        D[:, 3 * n:3 * n + 3] *= phase[n]

                    Q1 = int(round(np.dot(q, a1) * scale))
                    Q2 = int(round(np.dot(q, a2) * scale))

                    D_full[+Q1, +Q2] = D
                    D_full[-Q1, -Q2] = D.conj()

    if nq is None:
        nq = D_full.shape[:-2]
    elif not hasattr(nq, '__len__'):
        nq = (nq,) * len(q_irr[0])

    nq_orig = tuple(nq)
    nq = np.ones(3, dtype=int)
    nq[:len(nq_orig)] = nq_orig

    D_full = np.reshape(D_full, (nq[0], nq[1], nq[2], ph.size, ph.size))

    phid = np.fft.ifftn(D_full, axes=(0, 1, 2)).real
    phid = np.reshape(phid, (nq[0], nq[1], nq[2], ph.nat, 3, ph.nat, 3))
    phid = np.transpose(phid, (3, 5, 0, 1, 2, 4, 6))

    for na in range(ph.nat):
        phid[na, :] *= np.sqrt(ph.M[na])
        phid[:, na] *= np.sqrt(ph.M[na])

    if apply_asr_simple:
        asr(phid)

    ph.R, ph.data, ph.l = short_range_model(phid, ph.M, ph.a, ph.r,
        divide_mass=divide_mass, divide_ndegen=divide_ndegen)

    if apply_asr or apply_rsr:
        sum_rule_correction(ph, asr=apply_asr, rsr=apply_rsr)

def interpolate_dynamical_matrices(D, q, nq, fildyn_template, fildyn, flfrc,
        angle=120, write_fildyn0=True, apply_asr=False, apply_asr_simple=False,
        apply_rsr=False, qe_prefix='', clean=False):
    """Interpolate dynamical matrices given for irreducible wedge of q points.

    This function still uses the Quantum ESPRESSO executables ``q2qstar.x`` and
    ``q2r.x``.  They are called in serial by each MPI process, which leads to
    problems if they have been compiled for parallel execution. If you want to
    run this function in parallel, you have two choices:

        (1) Configure Quantum ESPRESSO for compilation of serial executables
            via ``./configure --disable-parallel`` and run ``make ph``. If you
            do not want to make them available through the environmental
            variable ``PATH``, you can also set the parameter `qe-prefix` to
            ``'/path/to/serial/q-e/bin/'``.  The trailing slash is required.

        (2) If your MPI implementation supports nested calls to ``mpirun``, you
            may try to set `qe_prefix` to ``'mpirun -np 1 '``. The trailing
            space is required.

    Parameters
    ----------
    D : list of square arrays
        Dynamical matrices for all irreducible q points.
    q : list of 2-tuples
        Irreducible q points in crystal coordinates with period :math:`2 \pi`.
    nq : int
        Number of q points per dimension, i.e., size of uniform mesh.
    fildyn_template : str
        Complete name of *fildyn* file from which to take header information.
    fildyn : str
        Prefix for written files with dynamical matrices.
    flfrc : str
        Name of written file with interatomic force constants.
    angle : float
        Angle between Bravais lattice vectors in degrees.
    write_fildyn0 : bool
        Write *fildyn0* needed by ``q2r.x``? Otherwise the file must be present.
    apply_asr : bool
        Enforce acoustic sum rule by overwriting self force constants?
    apply_asr_simple : bool
        Apply simple acoustic sum rule correction to force constants? This sets
        the self force constant to minus the sum of all other force constants.
    apply_rsr : bool
        Enforce rotation sum rule by overwriting self force constants?
    qe_prefix : str
        String to prepend to names of Quantum ESPRESSO executables.
    clean : bool
        Delete all temporary files afterwards?

    Returns
    -------
    function
        Fourier-interpolant (via force constants) for dynamical matrices.
    """
    import os

    # transform q points from crystal to Cartesian coordinates:

    a1, a2 = bravais.translations(angle)
    b1, b2 = bravais.reciprocals(a1, a2)

    q_cart = []

    for iq, (q1, q2) in enumerate(q):
        qx, qy = (q1 * b1 + q2 * b2) / (2 * np.pi)
        q_cart.append((qx, qy, 0.0))

    # write 'fildyn0' with information about q-point mesh:

    if write_fildyn0:
        write_q(fildyn + '0', q_cart, nq)

    # read 'fildyn' template and choose (arbitrary) footer text:

    if comm.rank == 0:
        data = read_fildyn(fildyn_template)
    else:
        data = None

    header, qpoints, dynmats, footer, amass = comm.bcast(data)
    footer = "File generated by 'elphmod' based on output from 'ph.x'"

    # write and complete 'fildyn1', 'fildyn2', ... with dynamical matrices:

    sizes, bounds = MPI.distribute(len(q), bounds=True)

    for iq in range(*bounds[comm.rank:comm.rank + 2]):
        fildynq = fildyn + str(iq + 1)

        write_fildyn(fildynq, header, [q_cart[iq]], [D[iq]], footer, amass,
            divide_mass=True)

        os.system('{0}q2qstar.x {1} {1} > /dev/null'.format(qe_prefix, fildynq))

    comm.Barrier()

    # compute interatomic force constants:

    if comm.rank == 0:
        os.system("""echo "&INPUT fildyn='{1}' flfrc='{2}' /" """
            '| {0}q2r.x > /dev/null'.format(qe_prefix, fildyn, flfrc))

    # clean up and return mass-spring model:
    # (no MPI barrier needed because of broadcasting in 'Model')

    ph = Model(flfrc, apply_asr=apply_asr, apply_asr_simple=apply_asr_simple,
        apply_rsr=apply_rsr)

    if clean:
        if comm.rank == 0:
            os.system('rm %s0 %s' % (fildyn, flfrc))

        for iq in range(*bounds[comm.rank:comm.rank + 2]):
            os.system('rm %s%d' % (fildyn, iq + 1))

    return ph
