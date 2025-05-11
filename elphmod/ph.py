# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

"""Mass-spring models from Quantum ESPRESSO."""

import numpy as np

import elphmod.bravais
import elphmod.dispersion
import elphmod.misc
import elphmod.MPI

comm = elphmod.MPI.comm
info = elphmod.MPI.info

class Model:
    r"""Mass-spring model for the phonons.

    Parameters
    ----------
    flfrc : str
        File with interatomic force constants from ``q2r.x`` or common filename
        part (without appended/inserted q-point number) of dynamical matrices
        from ``ph.x``. Both the plain-text and XML format are supported.
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
    amass : ndarray
        Atomic masses if `flfrc` is omitted.
    at : ndarray
        Bravais lattice vectors if `flfrc` is omitted.
    tau : ndarray
        Positions of basis atoms if `flfrc` is omitted.
    atom_order : list of str
        Ordered list of atoms if `flfrc` is omitted.
    alph : float
        Ewald parameter if `flfrc` is omitted.
    epsil : ndarray
        Dielectric tensor if `flfrc` is omitted.
    zeu : ndarray
        Born effective charges if `flfrc` is omitted.
    Q : ndarray
        Quadrupole tensors if `quadrupole_fmt` is omitted.
    L : float
        Range-separation parameter for two-dimensional electrostatics.
    perp : bool
        Yield out-of-plane long-range terms if `L` is nonzero?
    divide_mass : bool
        Divide force constants and Born effective charges by atomic masses?
    divide_ndegen : bool
        Divide force constants by degeneracy of Wigner-Seitz point? Only
        ``True`` yields correct phonons. ``False`` should only be used for
        debugging.
    ifc : str
        Name of file with interatomic force constants to be created if `flfrc`
        is prefix of files with dynamical matrices. Used to emulate ``q2r.x``.

    Attributes
    ----------
    R : ndarray
        Lattice vectors :math:`\vec R` of Wigner-Seitz supercell.
    data : ndarray
        Corresponding force constants divided by atomic masses in Ry\ :sup:`2`.

        .. math::

            D_{\vec R i j} = \frac{\hbar^2}{\sqrt{M_i M_j}}
                \frac{\partial^2 E}{\partial u_{0 i} \partial u_{\vec R j}}

        If not :attr:`divide_mass`, the prefactor :math:`\hbar^2 / \sqrt{M_i
        M_j}` is absent and the units are Ry/bohr\ :sup:`2` instead. If
        :attr:`lr`, this is only the short-range component of the force
        constants.
    M : ndarray
        Atomic masses :math:`M_i`.
    a : ndarray
        Bravais lattice vectors.
    r : ndarray
        Positions of basis atoms.
    l : ndarray.
        Bond lengths.
    atom_order : list of str
        Ordered list of atoms.
    alpha : float
        Ewald parameter.
    eps : ndarray
        Dielectric tensor.
    Z : ndarray
        Born effective charges.
    z : ndarray
        Born effective charges divided by square root of atomic masses.
    Q : ndarray
        Quadrupole tensors.
    q : ndarray
        Quadrupole tensors divided by square root of atomic masses.
    L : float
        Range-separation parameter for two-dimensional electrostatics.
    perp : bool
        Yield out-of-plane long-range terms if `L` is nonzero?
    divide_mass : bool
        Have force constants and Born charges been divided by atomic masses?
    divide_ndegen : bool
        Have force constants been divided by degeneracy of Wigner-Seitz point?
    size : int
        Number of displacement directions/bands.
    nat : int
        Number of atoms.
    nq : tuple of int
        Shape of original q-point mesh.
    q0 : ndarray
        Original q-point mesh.
    D0 : ndarray
        Dynamical matrices on original q-point mesh.
    cells : list of tuple of int, optional
        Lattice vectors of unit cells if the model describes a supercell.
    N : list of tuple of int, optional
        Primitive vectors of supercell if the model describes a supercell.
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
        r"""Set up dynamical matrix for arbitrary q point.

        Parameters
        ----------
        q1, q2, q3 : float, default 0.0
            q point in crystal coordinates with period :math:`2 \pi`.

        Returns
        -------
        ndarray
            Fourier transform of :attr:`data`, possibly plus a long-range term.
        """
        q = np.array([q1, q2, q3])

        # Sign convention in do_q2r.f90 of QE:
        # 231  CALL cfft3d ( phid (:,j1,j2,na1,na2), &
        # 232       nr1,nr2,nr3, nr1,nr2,nr3, 1, 1 )
        # 233  phid(:,j1,j2,na1,na2) = &
        # 234       phid(:,j1,j2,na1,na2) / DBLE(nr1*nr2*nr3)
        # The last argument of cfft3d is the sign (+1).

        Dq = np.einsum('Rxy,R->xy', self.data, np.exp(1j * self.R.dot(q)))

        if self.lr:
            Dq += self.D_lr(q1, q2, q3)

        return Dq

    def D_lr(self, q1=0, q2=0, q3=0):
        """Calculate long-range part of dynamical matrix."""

        factor, vector = self.generate_long_range(q1, q2, q3)

        return self.D0_lr + np.einsum('g,gi,gj->ij',
            factor, vector.conj(), vector)

    def C(self, R1=0, R2=0, R3=0):
        """Get interatomic force constants for arbitrary lattice vector.

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

    def __init__(self, flfrc=None, quadrupole_fmt=None, apply_asr=False,
            apply_asr_simple=False, apply_zasr=False, apply_rsr=False, lr=True,
            lr2d=None, amass=None, at=None, tau=None, atom_order=None,
            alph=None, epsil=None, zeu=None, Q=None, L=None, perp=True,
            divide_mass=True, divide_ndegen=True, ifc=None):

        phid = nq = q0 = D0 = None

        if comm.rank == 0:
            if flfrc is not None:
                try:
                    (phid, amass, at, tau, atom_order,
                        alph, epsil, zeu) = read_flfrc(flfrc)

                    nq = phid.shape[2:5]

                    if apply_asr_simple:
                        asr(phid)

                except FileNotFoundError:
                    xml = flfrc.lower().endswith('.xml')

                    if xml:
                        try:
                            nq, q0 = read_q('%s0' % flfrc[:-4])
                        except FileNotFoundError:
                            nq, q0 = read_q('%s0.xml' % flfrc[:-4])
                    else:
                        nq, q0 = read_q('%s0' % flfrc)

                    for iq0 in range(len(q0)):
                        if xml:
                            fildyn = ('%s%d%s'
                                % (flfrc[:-4], iq0 + 1, flfrc[-4:]))
                        else:
                            fildyn = '%s%d' % (flfrc, iq0 + 1)

                        if iq0:
                            q, D = read_flfrc(fildyn)[0]
                        else:
                            ((q, D), amass, at, tau, atom_order,
                                alph, epsil, zeu) = read_flfrc(fildyn)

                            q0 = np.empty((*nq, 3), dtype=float)
                            D0 = np.empty((*nq, 3 * len(amass), 3 * len(amass)),
                                dtype=complex)

                        if divide_mass:
                            divide_by_mass(D, amass)

                        for iq in range(len(q)):
                            q[iq] = np.dot(at, q[iq])

                            i = tuple(np.round(q[iq]
                                * nq / (2 * np.pi)).astype(int) % nq)

                            q0[i] = q[iq]
                            D0[i] = D[iq]

                    q0 = q0.reshape((-1, *q0.shape[3:]))
                    D0 = D0.reshape((-1, *D0.shape[3:]))

            if quadrupole_fmt is not None:
                Q = read_quadrupole_fmt(quadrupole_fmt)

            if apply_zasr and zeu is not None:
                zasr(zeu)

        phid = comm.bcast(phid)
        self.nq = comm.bcast(nq)
        self.q0 = comm.bcast(q0)
        self.D0 = comm.bcast(D0)

        self.M = comm.bcast(amass)
        self.a = comm.bcast(at)
        self.r = comm.bcast(tau)
        self.atom_order = comm.bcast(atom_order)

        self.alpha = comm.bcast(alph)
        self.eps = comm.bcast(epsil)
        self.Z = comm.bcast(zeu)
        self.Q = comm.bcast(Q)
        self.L = comm.bcast(L)
        self.perp = comm.bcast(perp)

        self.divide_mass = divide_mass
        self.divide_ndegen = divide_ndegen

        if self.atom_order is None:
            self.nat = self.size = None
        else:
            self.nat = len(self.atom_order)
            self.size = 3 * self.nat

        self.cells = [(0, 0, 0)]

        if self.alpha is None:
            self.alpha = 1.0

        self.lr = lr and self.eps is not None and self.Z is not None

        self.lr2d = (self.nq[0] != self.nq[2] == 1 != self.nq[1]
            if lr2d is None and self.nq is not None else lr2d)

        if self.lr:
            if self.lr2d != lr2d:
                info('Warning: System is assumed to be %s-dimensional!'
                    % ('two' if self.lr2d else 'three'))

            self.prepare_long_range()

        if phid is not None:
            self.R, self.data, self.l = elphmod.bravais.short_range_model(
                phid, self.a, self.r, sgn=-1, divide_ndegen=divide_ndegen)

            if divide_mass:
                divide_by_mass(self.data, self.M)

        elif self.D0 is not None:
            self.update_short_range(flfrc=ifc,
                apply_asr_simple=apply_asr_simple)

        else:
            return

        if apply_asr or apply_rsr:
            sum_rule_correction(self, asr=apply_asr, rsr=apply_rsr)

    def prepare_long_range(self, G_max=28.0, G_2d=False):
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
        * Improved 2D phonons: Royo and Stengel, Phys. Rev. X 11, 041027 (2021)
        * Improved 2D coupling: Ponce' et al., Phys. Rev. B 107, 155424 (2023)

        Parameters
        ----------
        G_max : float
            Cutoff for reciprocal lattice vectors.
        G_2d : bool, default False
            Do not sample out-of-plane reciprocal lattice vectors regardless of
            :attr:`lr2d`?
        """
        self.b = np.array(elphmod.bravais.reciprocals(*self.a))

        e2 = 2.0 # square of electron charge in Rydberg units

        if self.lr2d:
            c = 1 / self.b[2, 2]

            self.r_eff = (self.eps - np.eye(3)) * c / 2

            area = np.linalg.norm(np.cross(self.a[0], self.a[1]))
            self.prefactor = 2 * np.pi * e2 / area
        else:
            volume = abs(np.dot(self.a[0], np.cross(self.a[1], self.a[2])))
            self.prefactor = 4 * np.pi * e2 / volume

        a = np.linalg.norm(self.a[0])

        self.scale = 4 * self.alpha * (2 * np.pi / a) ** 2

        nr = 1 + (np.sqrt(self.scale * G_max)
            / np.linalg.norm(2 * np.pi * self.b, axis=1)).astype(int)

        if self.lr2d or G_2d:
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

        self.G = np.array(self.G)

        self.z = np.copy(self.Z)
        self.q = np.copy(self.Q)

        self.D0_lr = np.zeros((self.size, self.size), dtype=complex)
        Dq0_lr = self.D_lr()

        for na1 in range(self.nat):
            for na2 in range(self.nat):
                self.D0_lr[group(na1), group(na1)] -= (
                    Dq0_lr[group(na1), group(na2)])

        if self.divide_mass:
            for na in range(self.nat):
                self.D0_lr[group(na), group(na)] /= self.M[na]

                self.z[na] /= np.sqrt(self.M[na])

                if self.Q is not None:
                    self.q[na] /= np.sqrt(self.M[na])

    def generate_long_range(self, q1=0, q2=0, q3=0, perp=None, eps=1e-10):
        r"""Generate long-range terms.

        Parameters
        ----------
        q : ndarray
            q point in reciprocal lattice units :math:`q_i \in [0, 2 \pi)`.
        perp : bool
            Yield out-of-plane terms? Defaults to attribute :attr:`perp`.
        eps : float
            Tolerance for vanishing lattice vectors.

        Returns
        -------
        ndarray
            Scalar prefactor for relevant reciprocal lattice vectors.
        ndarray
            Direction-dependent term for relevant reciprocal lattice vectors.
        """
        if perp is None:
            perp = self.perp

        K = self.G + q1 * self.b[0] + q2 * self.b[1] + q3 * self.b[2]

        if self.lr2d:
            KeK = (K ** 2).sum(axis=1)
        else:
            KeK = np.einsum('gx,xy,gy->g', K, self.eps, K)

        use = KeK > eps

        KeK = KeK[use]
        K = K[use]

        if self.lr2d:
            KrK = np.einsum('gx,xy,gy->g',
                K[:, :2], self.r_eff[:2, :2], K[:, :2])

        if self.lr2d and self.L is not None:
            K_norm = np.sqrt(KeK)
            f = 1 - np.tanh(K_norm * self.L / 2)

            factor = self.prefactor * f / K_norm
            factor /= 1 + f / K_norm * KrK

            if perp:
                KrK_perp = self.r_eff[2, 2]

                factor_perp = -self.prefactor * f / K_norm
                factor_perp /= 1 - f * K_norm * KrK_perp
        else:
            factor = self.prefactor * np.exp(-KeK / self.scale)

            if self.lr2d:
                factor /= np.sqrt(KeK) + KrK
            else:
                factor /= KeK

        exp = np.exp(-1j * np.einsum('gx,nx->gn', K, self.r))
        exp = exp[:, :, np.newaxis]

        dot = 1j * np.einsum('ge,neu->gnu', K, self.z) # we use zeu, not zue

        if self.Q is not None:
            dot += 0.5 * np.einsum('gx,nuxy,gy->gnu', K, self.q, K)

            if self.lr2d and self.L is not None:
                dot -= 0.5 * np.einsum('g,nu->gnu',
                    K_norm ** 2, self.q[:, :, 2, 2])

        vector = np.reshape(dot * exp, (len(K), self.size))

        if self.lr2d and self.L is not None and perp:
            dot_perp = 1j * np.einsum('g,nu->gnu', K_norm, self.z[:, 2, :])

            if self.Q is not None:
                dot_perp += np.einsum('g,nuy,gy->gnu',
                    K_norm, self.q[:, :, 2, :2], K[:, :2])

            vector_perp = np.reshape(dot_perp * exp, (len(K), self.size))

            factor = np.concatenate((factor, factor_perp))
            vector = np.concatenate((vector, vector_perp))

        return factor, vector

    def sample_orig(self):
        """Sample dynamical matrix on original q-point mesh."""

        self.q0 = elphmod.bravais.mesh(*self.nq, flat=True)

        self.D0 = elphmod.dispersion.sample(self.D, self.q0)

    def update_short_range(self, flfrc=None, apply_asr_simple=False):
        """Update short-range part of interatomic force constants.

        Parameters
        ----------
        flfrc : str
            Filename where short-range force constants are written.
        apply_asr_simple : bool
            Apply simple acoustic sum rule correction to short-range force
            constants?
        """
        if self.D0 is None:
            info('Run "sample_orig" before changing Z, Q, etc.!', error=True)

        if not self.lr:
            q2r(self, nq=self.nq, D_full=self.D0, flfrc=flfrc,
                apply_asr_simple=apply_asr_simple, divide_mass=self.divide_mass)
            return

        self.prepare_long_range()

        D = self.D0 - elphmod.dispersion.sample(self.D_lr, self.q0)

        q2r(self, nq=self.nq, D_full=D, flfrc=flfrc,
            apply_asr_simple=apply_asr_simple, divide_mass=self.divide_mass)

    def sum_force_constants(self):
        """Calculate sum of absolute values of short-range force constants.

        For the optimal range-separation parameter this value is minimal.
        """
        C_all = abs(self.data).reshape(-1, self.nat, 3, self.nat, 3)
        C_all = C_all.sum(axis=(0, 2, 4))

        C_self = abs(self.C()).reshape(self.nat, 3, self.nat, 3)
        C_self = np.diag(np.diag(C_self.sum(axis=(1, 3))))

        cost = C_all - C_self

        if self.divide_mass:
            for na in range(self.nat):
                cost[na, :] *= np.sqrt(self.M[na])
                cost[:, na] *= np.sqrt(self.M[na])

        return cost.sum() / (2 * np.prod(self.nq))

    def supercell(self, N1=1, N2=1, N3=1, sparse=False):
        """Map mass-spring model onto supercell.

        Parameters
        ----------
        N1, N2, N3 : tuple of int or int, default 1
            Supercell lattice vectors in units of primitive lattice vectors.
        sparse : bool, default False
            Only calculate q = 0 dynamical matrix as a sparse matrix to save
            memory? The result is stored in the attribute :attr:`Ds`. Consider
            using :meth:`standardize` with nonzero `eps` and `symmetrize`
            before.

        Returns
        -------
        object
            Mass-spring model for supercell.

        See Also
        --------
        elphmod.bravais.supercell
        """
        ph = Model()

        supercell = elphmod.bravais.supercell(N1, N2, N3)
        ph.N = list(map(tuple, supercell[1]))
        ph.cells = supercell[-1]

        ph.M = np.tile(self.M, len(ph.cells))
        ph.a = np.dot(ph.N, self.a)
        ph.atom_order = list(self.atom_order) * len(ph.cells)
        ph.size = self.size * len(ph.cells)
        ph.nat = self.nat * len(ph.cells)
        ph.r = np.array([
            n1 * self.a[0] + n2 * self.a[1] + n3 * self.a[2] + self.r[na]
            for n1, n2, n3 in ph.cells
            for na in range(self.nat)])
        ph.divide_mass = self.divide_mass
        ph.divide_ndegen = self.divide_ndegen

        ph.lr = self.lr
        ph.lr2d = self.lr2d
        ph.L = self.L
        ph.perp = self.perp
        ph.eps = self.eps

        if self.alpha is None:
            ph.alpha = self.alpha
        else:
            ph.alpha = self.alpha * (np.linalg.norm(ph.a[0])
                / np.linalg.norm(self.a[0])) ** 2

        if self.Z is None:
            ph.Z = None
        else:
            ph.Z = np.tile(self.Z, (len(ph.cells), 1, 1))

        if self.Q is None:
            ph.Q = None
        else:
            ph.Q = np.tile(self.Q, (len(ph.cells), 1, 1, 1))

        if self.lr:
            ph.prepare_long_range()

        if sparse:
            sparse_array = elphmod.misc.get_sparse_array()

            if ph.lr:
                ph.Ds = sparse_array(ph.D_lr().real)
            else:
                ph.Ds = sparse_array((ph.size, ph.size))

        if comm.rank == 0:
            const = dict()

            status = elphmod.misc.StatusBar(len(ph.cells),
                title='map force constants onto supercell')

            for i in range(len(ph.cells)):
                X = i * self.size

                for n in range(len(self.R)):
                    R, r = elphmod.bravais.to_supercell(self.R[n] + ph.cells[i],
                        supercell)

                    Y = r * self.size

                    if sparse:
                        ph.Ds[X:X + self.size, Y:Y + self.size] += self.data[n]
                        continue

                    if R not in const:
                        const[R] = np.zeros((ph.size, ph.size))

                    const[R][X:X + self.size, Y:Y + self.size] = self.data[n]

                status.update()

            ph.R = np.array(list(const.keys()), dtype=int).reshape((-1, 3))
            ph.data = np.array(list(const.values())).reshape((-1,
                ph.size, ph.size))

            count = len(const)
            const.clear()
        else:
            count = None

        if sparse:
            ph.Ds = comm.bcast(ph.Ds)

            return ph

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

        ph.divide_mass = self.divide_mass
        ph.divide_ndegen = self.divide_ndegen

        ph.lr = self.lr
        ph.lr2d = self.lr2d
        ph.L = self.L
        ph.perp = self.perp
        ph.eps = self.eps

        if self.alpha is None:
            ph.alpha = self.alpha
        else:
            ph.alpha = self.alpha * (np.linalg.norm(ph.a[0])
                / np.linalg.norm(self.a[0])) ** 2

        if self.Z is None:
            ph.Z = None
        else:
            ph.Z = self.Z[:ph.nat]

        if self.Q is None:
            ph.Q = None
        else:
            ph.Q = self.Q[:ph.nat]

        if self.lr:
            ph.prepare_long_range()

        if comm.rank == 0:
            const = dict()

            status = elphmod.misc.StatusBar(len(self.R),
                title='map force constants back to unit cell')

            for n in range(len(self.R)):
                for i, cell in enumerate(self.cells):
                    C = self.data[n, :ph.size, i * ph.size:(i + 1) * ph.size]

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
        order = list(order)

        self.data = self.data.reshape((len(self.R), self.nat, 3, self.nat, 3))

        self.data = self.data[:, order, :, :, :]
        self.data = self.data[:, :, :, order, :]

        self.data = self.data.reshape((len(self.R), self.size, self.size))

        self.M = np.array([self.M[na] for na in order])
        self.atom_order = [self.atom_order[na] for na in order]
        self.r = np.array([self.r[na] for na in order])

        if self.Z is not None:
            self.Z = self.Z[order]

        if self.Q is not None:
            self.Q = self.Q[order]

        if self.lr:
            self.prepare_long_range()

    def shift_atoms(self, s, S):
        """Move selected atoms across unit-cell boundary.

        Together with :func:`order_atoms`, this function helps reconcile
        inconsistent definitions of the basis/motif of the Bravais lattice.

        Parameters
        ----------
        s : slice
            Slice of atom indices corresponding to selected basis atom(s).
        S : tuple of int
            Shift of as multiple of primitive lattice vectors.
        """
        self.data = self.data.reshape((len(self.R), self.nat, 3, self.nat, 3))

        R = np.tile(self.R, (3, 1))
        data = np.tile(self.data, (3, 1, 1, 1, 1))

        n = len(self.R)
        m = 2 * n

        R[:n] -= S
        data[:n, :, :, :, :] = 0.0
        data[:n, :, :, s, :] = self.data[:, :, :, s, :]
        data[:n, s, :, s, :] = 0.0

        data[n:m, s, :, :, :] = 0.0
        data[n:m, :, :, s, :] = 0.0
        data[n:m, s, :, s, :] = self.data[:, s, :, s, :]

        R[m:] += S
        data[m:, :, :, :, :] = 0.0
        data[m:, s, :, :, :] = self.data[:, s, :, :, :]
        data[m:, s, :, s, :] = 0.0

        self.R = R
        self.data = data

        self.data = self.data.reshape((len(self.R), self.size, self.size))

        self.standardize()

        self.r[s] += np.dot(S, self.a)

    def standardize(self, eps=0.0, symmetrize=False):
        r"""Standardize mass-spring data.

        - Keep only nonzero force-constant matrices.
        - Sum over repeated lattice vectors.
        - Sort lattice vectors.
        - Optionally symmetrize force constants:

        .. math::

            D_{\vec q} = D_{\vec q}^\dagger,
            D_{\vec R} = D_{-\vec R}^\dagger
        """
        if comm.rank == 0:
            if eps:
                self.data[abs(self.data) < eps * abs(self.data).max()] = 0.0

            const = dict()

            status = elphmod.misc.StatusBar(len(self.R),
                title='standardize mass-spring data')

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
                            const[R] += self.data[n].T
                        else:
                            const[R] = self.data[n].T.copy()

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
            self.data = np.empty((count, self.size, self.size))

        comm.Bcast(self.R)
        comm.Bcast(self.data)

    def symmetrize(self):
        """Symmetrize dynamical matrix."""

        self.standardize(symmetrize=True)

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

    def clear(self):
        """Delete all lattice vectors and associated matrix elements."""

        self.R = np.zeros_like(self.R[:0, :])
        self.data = np.zeros_like(self.data[:0, :, :])

    def to_pwi(self, pwi, **kwargs):
        """Save atomic positions etc. to PWscf input file.

        Parameters
        ----------
        pwi : str
            Filename.
        **kwargs
            Keyword arguments with further parameters to be written.
        """
        species = sorted(set(self.atom_order),
            key=lambda X: self.atom_order.index(X))

        alat = np.linalg.norm(self.a[0])

        pw = dict()

        pw['ibrav'] = 0
        pw['ntyp'] = len(species)
        pw['nat'] = self.nat
        pw['celldm'] = [alat]

        pw['at_species'] = species
        pw['mass'] = [self.M[self.atom_order.index(X)] / elphmod.misc.uRy
            for X in species]
        pw['pp'] = ['%s.upf' % X for X in species]

        pw['coords'] = 'crystal'
        pw['at'] = self.atom_order
        pw['r'] = elphmod.bravais.cartesian_to_crystal(self.r, *self.a)

        pw['cell_units'] = 'alat'
        pw['r_cell'] = self.a / alat

        pw.update(kwargs)

        elphmod.bravais.write_pwi(pwi, pw)

    def to_flfrc(self, flfrc, nr1=None, nr2=None, nr3=None):
        """Save mass-spring model to force-constants file.

        Parameters
        ----------
        flfrc : str
            Filename.
        nr1, nr2, nr3 : int, optional
            Mesh dimensions. If not given, :attr:`nq` is assumed.
        """
        if nr1 is None:
            nr1 = self.nq[0]

        if nr2 is None:
            nr2 = self.nq[1]

        if nr3 is None:
            nr3 = self.nq[2]

        phid = np.zeros((self.nat, self.nat, nr1, nr2, nr3, 3, 3))

        sizes, bounds = elphmod.MPI.distribute(len(self.R), bounds=True)

        for n in range(*bounds[comm.rank:comm.rank + 2]):
            m1, m2, m3 = -self.R[n] % (nr1, nr2, nr3)

            phid[:, :, m1, m2, m3, :, :] += np.reshape(self.data[n],
                (self.nat, 3, self.nat, 3)).transpose(0, 2, 1, 3)

        comm.Reduce(elphmod.MPI.MPI.IN_PLACE if comm.rank == 0 else phid, phid)

        if comm.rank == 0:
            if self.divide_mass:
                for na in range(self.nat):
                    phid[na, :] *= np.sqrt(self.M[na])
                    phid[:, na] *= np.sqrt(self.M[na])

            if not self.lr and self.eps is not None and self.Z is not None:
                print('Warning: Writing inconsistent force-constants file!')
                print('The long-range part has not been subtracted (lr=False).')
                print('The dielectric properties are written nevertheless.')
                print('Consider setting the attribute Z to None to avoid this.')

            write_flfrc(flfrc, phid, self.M, self.a, self.r, self.atom_order,
                self.alpha, self.eps, self.Z)

def group(n, size=3):
    """Create slice of dynamical matrix belonging to `n`-th atom."""

    return slice(n * size, (n + 1) * size)

def divide_by_mass(dyn, amass):
    """Divide dynamical matrices by atomic masses.

    Parameters
    ----------
    dyn : ndarray
        Dynamical matrices.
    amass : list of float
        Atomic masses.
    """
    for na in range(len(amass)):
        dyn[..., group(na), :] /= np.sqrt(amass[na])
        dyn[..., :, group(na)] /= np.sqrt(amass[na])

def read_q(fildyn0):
    """Read list of irreducible q points from *fildyn0*."""

    with open(fildyn0) as data:
        nq = tuple(map(int, next(data).split()))
        nQ = int(next(data))
        q = [list(map(float, line.split())) for line in data]

    return nq, q

def write_q(fildyn0, q, nq):
    """Write list of irreducible q points to *fildyn0*."""

    with open(fildyn0, 'w') as data:
        data.write('%4d%4d%4d\n' % (nq, nq, 1))
        data.write('%4d\n' % len(q))

        for qxy in q:
            data.write('%19.15f%19.15f%19.15f\n' % (qxy[0], qxy[1], 0.0))

def fildyn_freq(fildyn='matdyn'):
    """Create *fildyn.freq* as created by Quantum ESPRESSO's ``ph.x``.

    Parameters
    ----------
    fildyn : str, default 'matdyn'
        Prefix of files with dynamical matrices.
    """
    if comm.rank != 0:
        return

    nq, q0 = read_q('%s0' % fildyn)

    with open('%s.freq' % fildyn, 'w') as freq:
        for iq in range(len(q0)):
            (q, D), amass, at, tau, atom_order, alph, epsil, zeu = read_flfrc(
                '%s%d' % (fildyn, iq + 1))

            divide_by_mass(D, amass)

            w = sgnsqrt(np.linalg.eigvalsh(D[0]))
            w *= elphmod.misc.Ry / elphmod.misc.cmm1

            if iq == 0:
                freq.write(' &plot nbnd=%4d, nks=%4d /\n' % (len(w), len(q0)))

            freq.write('%20.6f %9.6f %9.6f\n' % tuple(q0[iq]))

            for nu, wnu in enumerate(w, 1):
                freq.write(' %9.4f' % wnu)

                if not nu % 6 or nu == len(w):
                    freq.write('\n')

def read_flfrc(flfrc):
    """Read force constants in real or reciprocal space.

    `flfrc` can be either the force-constants file generated by ``q2r.x`` or one
    of the dynamical-matrix files generated by ``ph.x``.

    Returns
    -------
    ndarray or tuple of list of ndarray
        Force constants for input generated by ``q2r.x`` or equivalent q points
        and corresponding dynamical matrices for input generated by ``ph.x``.
    ndarray
        Atomic masses.
    ndarray
        Bravais lattice vectors.
    ndarray
        Atomic positions/basis vectors.
    list of str
        Atomic symbols.
    float
        Ewald parameter.
    ndarray
        Dielectric tensor.
    ndarray
        Born effective charges.

    See Also
    --------
    write_flfrc : Inverse function.
    """
    if flfrc.lower().endswith('.xml'):
        return read_flfrc_xml(flfrc)

    with open(flfrc) as data:
        # read all words of current line:

        def cells():
            for line in data:
                words = line.split()

                if words:
                    return words

        # read table:

        def table(rows):
            return np.array([list(map(float, cells())) for row in range(rows)])

        # detect file type:

        tmp = cells()

        dyn = tmp[0] == 'Dynamical'

        if dyn:
            next(data) # skip title
            tmp = cells()

        # read crystal structure:

        ntyp, nat, ibrav = list(map(int, tmp[:3]))
        celldm = list(map(float, tmp[3:]))

        # see Modules/latgen.f90 of Quantum ESPRESSO:

        if ibrav:
            at = elphmod.bravais.primitives(ibrav, celldm=celldm, bohr=True)
        else: # free
            if dyn:
                next(data) # skip "Basis vectors"

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

        amass = amass[ityp]

        atom_order = []

        for index in ityp:
            atom_order.append(atm[index].strip())

        alph = epsil = zeu = None

        if dyn:
            # read sections of dynamical-matrix file:

            q = []
            D = []

            while True:
                line = ' '.join(cells() or []).lower()

                if 'dynamical matrix in cartesian axes' in line:
                    q.append(np.array(list(map(float, cells()[3:6])))
                        * 2 * np.pi / celldm[0])

                    D.append(np.empty((3 * nat, 3 * nat), dtype=complex))

                    for na1 in range(nat):
                        for na2 in range(nat):
                            cells() # skip line with na1, na2

                            for j1 in range(3):
                                values = list(map(float, cells()))

                                for j2 in range(3):
                                    D[-1][group(na1), group(na2)][j1, j2] \
                                        = complex(*values[group(j2, 2)])

                elif 'dielectric tensor' in line:
                    epsil = table(3)

                elif 'effective charges e-u' in line:
                    zeu = np.empty((nat, 3, 3))

                    for na in range(nat):
                        na = int(cells()[-1]) - 1
                        zeu[na] = table(3)

                else:
                    break

            q = np.array(q)
            D = np.array(D)
        else:
            # read macroscopic dielectric function and effective charges:

            tmp = cells()

            lrigid = tmp[0] == 'T'

            if lrigid:
                if len(tmp) > 1:
                    alph = float(tmp[1])

                epsil = table(3)

                zeu = np.empty((nat, 3, 3))

                for na in range(nat):
                    na = int(cells()[0]) - 1
                    zeu[na] = table(3)

            # read interatomic force constants:

            nr1, nr2, nr3 = map(int, cells())

            phid = np.empty((nat, nat, nr1, nr2, nr3, 3, 3))

            for j1 in range(3):
                for j2 in range(3):
                    for na1 in range(nat):
                        for na2 in range(nat):
                            cells() # skip line with j1, j2, na1, na2

                            for m3 in range(nr3):
                                for m2 in range(nr2):
                                    for m1 in range(nr1):
                                        phid[na1, na2, m1, m2, m3, j1, j2] \
                                            = float(cells()[3])

    # return force constants, masses, and geometry:

    return [(q, D) if dyn else phid,
        amass, at, tau, atom_order, alph, epsil, zeu]

def write_flfrc(flfrc, phid, amass, at, tau, atom_order,
        alph=None, epsil=None, zeu=None):
    """Write force constants in real or reciprocal space.

    Parameters
    ----------
    flfrc : str
        Filename.
    phid : ndarray or tuple of list of ndarray
        Force constants or equivalent q points and corresponding dynamical
        matrices.
    amass : ndarray
        Atomic masses.
    at : ndarray
        Bravais lattice vectors.
    tau : ndarray
        Atomic positions/basis vectors.
    atom_order : list of str
        Atomic symbols.
    alph : float
        Ewald parameter.
    epsil : ndarray
        Dielectric tensor.
    zeu : ndarray
        Born effective charges.

    See Also
    --------
    read_flfrc : Inverse function.
    """
    dyn = isinstance(phid, tuple)

    atm, amass = tuple(zip(*sorted(set(zip(atom_order, amass)),
        key=lambda X: atom_order.index(X[0]))))

    with open(flfrc, 'w') as data:
        if dyn:
            data.write('Dynamical matrix file\n\n')

        # write crystal structure:

        data.write('%3d %4d %2d' % (len(atm), len(atom_order), 0))

        alat = np.linalg.norm(at[0])

        for celldm in alat, 0, 0, 0, 0, 0:
            data.write(' %10.7f' % celldm)

        data.write('\n')

        if dyn:
            data.write('Basis vectors\n')

        for x in range(3):
            data.write(' ')

            for y in range(3):
                data.write(' %14.9f' % (at[x, y] / alat))

            data.write('\n')

        # write palette of atomic species and masses:

        for nt in range(len(atm)):
            data.write("%12d  '%-3s' %19.10f\n" % (nt + 1, atm[nt], amass[nt]))

        # write types and positions of individual atoms:

        for na in range(len(tau)):
            data.write('%5d %4d' % (na + 1, atm.index(atom_order[na]) + 1))

            for x in range(3):
                data.write(' %17.10f' % (tau[na, x] / alat))

            data.write('\n')

        if dyn:
            q = phid[0] * alat / (2 * np.pi)
            D = np.array(phid[1])

            # write dynamical matrices:

            for iq in range(len(q)):
                data.write('\n     Dynamical matrix in Cartesian axes\n')
                data.write('\n     q = ( ')

                for coordinate in q[iq]:
                    data.write('%14.9f' % coordinate)

                data.write(' )\n\n')

                for na1 in range(len(atom_order)):
                    for na2 in range(len(atom_order)):
                        data.write('%5d%5d\n' % (na1 + 1, na2 + 1))
                        for j in range(3):
                            data.write('  '.join('%12.8f%12.8f'
                                % (z.real, z.imag)
                                    for z in D[iq, group(na1), group(na2)][j]))
                            data.write('\n')

        # write macroscopic dielectric function and effective charges:

        lrigid = epsil is not None and zeu is not None

        if not dyn:
            data.write('%2s' % ('T' if lrigid else 'F'))

            if lrigid and alph is not None:
                data.write(' %g' % alph)

            data.write('\n')

        if lrigid:
            if dyn:
                data.write('\n     Dielectric tensor\n')

            for x in range(3):
                for y in range(3):
                    data.write(' %23.12f' % epsil[x, y])

                data.write('\n')

            if dyn:
                data.write('\n     Effective charges\n')

            for na in range(len(zeu)):
                if dyn:
                    data.write('     atom #')

                data.write('%5d\n' % (na + 1))

                for x in range(3):
                    for y in range(3):
                        data.write(' %14.7f' % zeu[na, x, y])

                    data.write('\n')

        if not dyn:
            # write interatomic force constants:

            data.write('%4d %3d %3d\n' % phid.shape[2:5])

            for j1 in range(phid.shape[6]):
                for j2 in range(phid.shape[5]):
                    for na1 in range(phid.shape[0]):
                        for na2 in range(phid.shape[1]):
                            data.write('%4d %3d %3d %3d\n'
                                % (j1 + 1, j2 + 1, na1 + 1, na2 + 1))

                            for m3 in range(phid.shape[4]):
                                for m2 in range(phid.shape[3]):
                                    for m1 in range(phid.shape[2]):
                                        data.write('%4d %3d %3d %19.11E\n'
                                            % (m1 + 1, m2 + 1, m3 + 1, phid
                                                [na1, na2, m1, m2, m3, j1, j2]))

def read_flfrc_xml(flfrc):
    """Read force constants in real or reciprocal space from XML files.

    See Also
    --------
    read_flfrc : Equivalent function for non-XML files.
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.parse(flfrc).getroot()

    except ET.ParseError:
        with open(flfrc) as data:
            xml = data.read()

            # handle QE 7.2 XML issue (https://gitlab.com/QEF/q-e/-/issues/582):

            xml = xml.replace('</root>', '</Root>')

            if not xml.rstrip().endswith('</Root>'):
                xml += '</Root>'

            root = ET.fromstring(xml)

    geometry = root.find('GEOMETRY_INFO')

    ntyp = int(geometry.find('NUMBER_OF_TYPES').text)
    nat = int(geometry.find('NUMBER_OF_ATOMS').text)
    ibrav = int(geometry.find('BRAVAIS_LATTICE_INDEX').text)

    celldm = list(map(float, geometry.find('CELL_DIMENSIONS').text.split()))

    if ibrav:
        at = elphmod.bravais.primitives(ibrav, celldm=celldm, bohr=True)
    else:
        at = list(map(float, geometry.find('AT').text.split()))
        at = np.array(at).reshape((3, 3)) * celldm[0]

    atm = []
    amass = np.empty(ntyp)

    for nt in range(ntyp):
        atm.append(geometry.find('TYPE_NAME.%d' % (nt + 1)).text)
        amass[nt] = float(geometry.find('MASS.%d' % (nt + 1)).text)

    ityp = np.empty(nat, dtype=int)
    tau = np.empty((nat, 3))

    for na in range(nat):
        atom = geometry.find('ATOM.%d' % (na + 1))

        ityp[na] = int(atom.attrib['INDEX']) - 1
        tau[na, :] = list(map(float, atom.attrib['TAU'].split()))

    tau *= celldm[0]

    amass = amass[ityp] * elphmod.misc.uRy

    atom_order = []

    for index in ityp:
        atom_order.append(atm[index])

    dielect = root.find('DIELECTRIC_PROPERTIES')

    epsil = dielect.find('EPSILON')

    if epsil is not None:
        epsil = np.reshape(list(map(float, epsil.text.split())), (3, 3))

    zeu = dielect.find('ZSTAR')

    if zeu is not None:
        zeu = np.reshape([list(map(float, zeu.find('Z_AT_.%d'
            % (na + 1)).text.split())) for na in range(nat)], (nat, 3, 3))

    ifc = root.find('INTERATOMIC_FORCE_CONSTANTS')

    if ifc is not None:
        alph = ifc.find('alpha_ewald')

        if alph is not None:
            alph = float(alph.text)

        nr1, nr2, nr3 = map(int, ifc.find('MESH_NQ1_NQ2_NQ3').text.split())

        phid = np.empty((nat, nat, nr1, nr2, nr3, 3, 3))

        for na1 in range(nat):
            for na2 in range(nat):
                for m3 in range(nr3):
                    for m2 in range(nr2):
                        for m1 in range(nr1):
                            tmp = ifc.find('s_s1_m1_m2_m3.%d.%d.%d.%d.%d'
                                % (na1 + 1, na2 + 1, m1 + 1, m2 + 1, m3 + 1))

                            tmp = list(map(float, tmp.find('IFC').text.split()))

                            tmp = np.reshape(tmp, (3, 3)).T

                            phid[na1, na2, m1, m2, m3] = tmp
    else:
        alph = None

        nq = int(geometry.find('NUMBER_OF_Q').text)

        q = np.empty((nq, 3))
        D = np.empty((nq, 3 * nat, 3 * nat), dtype=complex)

        for iq in range(nq):
            dynmat = root.find('DYNAMICAL_MAT_.%d' % (iq + 1))

            q[iq] = list(map(float, dynmat.find('Q_POINT').text.split()))
            q[iq] *= 2 * np.pi / celldm[0]

            for na1 in range(nat):
                for na2 in range(nat):
                    tmp = np.array(list(map(float, dynmat.find('PHI.%d.%d'
                        % (na1 + 1, na2 + 1)).text.split())))

                    tmp = np.reshape(tmp[0::2] + 1j * tmp[1::2], (3, 3)).T

                    D[iq, group(na1), group(na2)] = tmp

    return [phid if ifc else (q, D),
        amass, at, tau, atom_order, alph, epsil, zeu]

def read_quadrupole_fmt(quadrupole_fmt):
    """Read file *quadrupole.fmt* suitable for ``epw.x``.

    Parameters
    ----------
    quadrupole_fmt : str
        Filename.

    Returns
    -------
    Q : ndarray
        Quadrupole tensor.

    See Also
    --------
    write_quadrupole_fmt : Inverse function.
    """
    Q = []

    with open(quadrupole_fmt) as data:
        next(data)

        for line in data:
            cols = line.split()

            if not cols:
                continue

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

def write_quadrupole_fmt(quadrupole_fmt, Q):
    """Write file *quadrupole.fmt* suitable for ``epw.x``.

    Parameters
    ----------
    quadrupole_fmt : str
        Filename.
    Q : ndarray
        Quadrupole tensor.

    See Also
    --------
    read_quadrupole_fmt : Inverse function.
    """
    with open(quadrupole_fmt, 'w') as data:
        data.write(('%4s %3s' + ' %14s' * 6 + '\n')
            % ('atom', 'dir', 'Qxx', 'Qyy', 'Qzz', 'Qyz', 'Qxz', 'Qxy'))

        for na in range(Q.shape[0]):
            for i in range(Q.shape[1]):
                data.write(('%4d %3d' + ' %14.10f' * 6 + '\n') % (na + 1, i + 1,
                    Q[na, i, 0, 0], Q[na, i, 1, 1], Q[na, i, 2, 2],
                    Q[na, i, 1, 2], Q[na, i, 0, 2], Q[na, i, 0, 1]))

def asr(phid):
    """Apply simple acoustic sum rule correction to force constants."""

    nat, nr1, nr2, nr3 = phid.shape[1:5]

    for na1 in range(nat):
        phid[na1, na1, 0, 0, 0] = -sum(phid[na1, na2, m1, m2, m3]
            for na2 in range(nat)
            for m1 in range(nr1)
            for m2 in range(nr2)
            for m3 in range(nr3)
            if na1 != na2 or m1 or m2 or m3)

def zasr(Z):
    """Apply acoustic sum rule correction to Born effective charges."""

    Z -= np.average(Z, axis=0)

def sum_rule_correction(ph, asr=True, rsr=True, eps=1e-15, report=True):
    """Apply sum rule correction to force constants.

    Unlike :func:`asr` called by the argument `apply_asr_simple`, the corrected
    force constants may not fulfill the point-symmetries of the crystal anymore.
    In turn, the total deviation from the original force constants is minimized.

    Parameters
    ----------
    ph : :class:`Model`
        Mass-spring model for the phonons.
    asr : bool
        Enforce acoustic sum rule?
    rsr : bool
        Enforce Born-Huang rotation sum rule?
    eps : float
        Smallest safe absolute value of divisor.
    report : bool
        Print sums before and after correction?
    """
    if comm.rank != 0:
        comm.Bcast(ph.data)
        return

    if report:
        print('Warning: Sum-rule corrections do not respect point symmetries!')
        print('Consider simple ASR correction instead (apply_asr_simple).')

    # define sums that should be zero:

    def acoustic_sum():
        zero = 0.0
        for k in range(ph.nat):
            for x in range(3):
                for y in range(3):
                    S = 0.0
                    for n in range(len(R)):
                        for l in range(ph.nat):
                            S += C[n, k, x, l, y]
                    zero += abs(S)
        return zero

    def rotation_sum():
        zero = 0.0
        for k in range(ph.nat):
            for x1 in range(3):
                for x2 in range(3):
                    for y in range(3):
                        S = 0.0
                        for n in range(len(R)):
                            for l in range(ph.nat):
                                S += (C[n, k, x1, l, y]
                                    * (R[n, x2] + ph.r[l, x2]))
                                S -= (C[n, k, x2, l, y]
                                    * (R[n, x1] + ph.r[l, x1]))
                        zero += abs(S)
        return zero

    # prepare lattice vectors and force constants:

    R = np.einsum('xy,nx->ny', ph.a, ph.R)
    C = ph.data.reshape((len(R), ph.nat, 3, ph.nat, 3))

    if ph.divide_mass:
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
        for k in range(ph.nat):
            for x in range(3):
                for y in range(3):
                    c.append(np.zeros_like(C))
                    for n in range(len(R)):
                        for l in range(ph.nat):
                            c[-1][n, k, x, l, y] += 1.0

    if rsr:
        for k in range(ph.nat):
            for x1 in range(3):
                for x2 in range(3):
                    for y in range(3):
                        c.append(np.zeros_like(C))
                        for n in range(len(R)):
                            for l in range(ph.nat):
                                c[-1][n, k, x1, l, y] += R[n, x2] + ph.r[l, x2]
                                c[-1][n, k, x2, l, y] -= R[n, x1] + ph.r[l, x1]

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

    if ph.divide_mass:
        for na in range(ph.nat):
            C[:, na, :, :, :] /= np.sqrt(ph.M[na])
            C[:, :, :, na, :] /= np.sqrt(ph.M[na])

    comm.Bcast(ph.data)

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

    a1, a2 = elphmod.bravais.translations(180 - angle)
    b1, b2 = elphmod.bravais.reciprocals(a1, a2)

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
        apply_asr=False, apply_asr_simple=False, apply_rsr=False, flfrc=None,
        divide_mass=True):
    r"""Interpolate dynamical matrices given for irreducible wedge of q points.

    This function replaces `interpolate_dynamical_matrices`, which depends on
    Quantum ESPRESSO. For 2D lattices, it is sufficient to provide dynamical
    matrices `D_irr` for the irreducible q points `q_irr`. Here, for the square
    lattice, the rotation symmetry (90 degrees) is currently disabled! In turn,
    for 1D and 2D lattices, dynamical matrices `D_full` on the complete uniform
    q-point mesh must be given.

    Parameters
    ----------
    ph : :class:`Model`
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
    flfrc : str
        Filename. If given, save file with force constants as generated by
        ``q2r.x``. This is done before acoustic and rotation sum are applied.
    divide_mass : bool
        Have input dynamical matrices been divided by atomic mass? This is
        independent of ``ph.divide_mass``, which is always respected.
    """
    if D_full is None:
        ph.size = D_irr.shape[-2]
        ph.nat = ph.size // 3

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

        a1, a2 = elphmod.bravais.translations(180 - angle)
        b1, b2 = elphmod.bravais.reciprocals(a1, a2)

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
                        D[group(n), :] *= phase[n].conj()
                        D[:, group(n)] *= phase[n]

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

    ph.size = D_full.shape[-2]
    ph.nat = ph.size // 3

    D_full = np.reshape(D_full, (*nq, ph.size, ph.size))

    phid = np.fft.ifftn(D_full, axes=(0, 1, 2)).real
    phid = np.reshape(phid, (*nq, ph.nat, 3, ph.nat, 3))
    phid = np.transpose(phid, (3, 5, 0, 1, 2, 4, 6))

    if divide_mass:
        for na in range(ph.nat):
            phid[na, :] *= np.sqrt(ph.M[na])
            phid[:, na] *= np.sqrt(ph.M[na])

    if flfrc and comm.rank == 0:
        write_flfrc(flfrc, phid, ph.M, ph.a, ph.r, ph.atom_order,
            ph.alpha, ph.eps, ph.Z)

    if apply_asr_simple:
        asr(phid)

    if ph.L is not None:
        S = abs(phid).sum() - np.trace(abs(phid))[0, 0, 0].sum() / np.prod(nq)

        info('Sum of force constants: %g Ry/bohr^2 (L = %g bohr)' % (S, ph.L))

    ph.R, ph.data, ph.l = elphmod.bravais.short_range_model(phid, ph.a, ph.r,
        sgn=-1, divide_ndegen=ph.divide_ndegen)

    if ph.divide_mass:
        divide_by_mass(ph.data, ph.M)

    ph.nq = tuple(nq)

    if apply_asr or apply_rsr:
        sum_rule_correction(ph, asr=apply_asr, rsr=apply_rsr)

def interpolate_dynamical_matrices(D, q, nq, fildyn_template, fildyn, flfrc,
        write_fildyn0=True, apply_asr=False, apply_asr_simple=False,
        apply_rsr=False, divide_mass=True, qe_prefix='', clean=False):
    r"""Interpolate dynamical matrices given for irreducible wedge of q points.

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
    write_fildyn0 : bool
        Write *fildyn0* needed by ``q2r.x``? Otherwise the file must be present.
    apply_asr : bool
        Enforce acoustic sum rule by overwriting self force constants?
    apply_asr_simple : bool
        Apply simple acoustic sum rule correction to force constants? This sets
        the self force constant to minus the sum of all other force constants.
    apply_rsr : bool
        Enforce rotation sum rule by overwriting self force constants?
    divide_mass : bool
        Have input dynamical matrices been divided by atomic mass? This is
        independent of ``ph.divide_mass``, which is always respected.
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

    # read 'fildyn' template:

    if comm.rank == 0:
        data = read_flfrc(fildyn_template)[1:]
    else:
        data = None

    amass, at, tau, atom_order, alph, epsil, zeu = comm.bcast(data)

    # transform q points from crystal to Cartesian coordinates:

    b1, b2, b3 = elphmod.bravais.reciprocals(*at)

    q_cart = []

    for iq, (q1, q2) in enumerate(q):
        q_cart.append(q1 * b1 + q2 * b2)

    # write 'fildyn0' with information about q-point mesh:

    if write_fildyn0:
        write_q(fildyn + '0', q_cart, nq)

    # write and complete 'fildyn1', 'fildyn2', ... with dynamical matrices:

    if divide_mass:
        D = np.copy(D)

        for na in range(len(amass)):
            D[:, group(na), :] *= np.sqrt(amass[na])
            D[:, :, group(na)] *= np.sqrt(amass[na])

    sizes, bounds = elphmod.MPI.distribute(len(q), bounds=True)

    for iq in range(*bounds[comm.rank:comm.rank + 2]):
        fildynq = fildyn + str(iq + 1)

        Gamma = np.all(q_cart[iq] == 0)

        write_flfrc(fildynq, (q_cart[iq:iq + 1], D[iq:iq + 1]), amass, at, tau,
            atom_order, alph, epsil if Gamma else None, zeu if Gamma else None)

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

def spectral_function(D, omega, eta):
    """Calculate phonon spectral function.

    See Eq. 76 by Monacelli et al., J. Phys. Condens. Matter 33, 363001 (2021).
    We use an additional prefactor of 2 to normalize the frequency integral
    (from zero to infinity) of the phonon spectral function to the number of
    modes (for each q point).

    Parameters
    ----------
    D : ndarray
        Dynamical matrix. The last three axes belong to the two displacement
        directions and the frequency argument.
    omega : ndarray
        Frequency arguments.
    eta : float
        Broadening parameter for all phonon modes.

    Returns
    -------
    ndarray
        Spectral function. The first axes belongs to the frequency argument.
    """
    sizes, bounds = elphmod.MPI.distribute(len(omega), bounds=True)

    my_A = np.empty((sizes[comm.rank], *D.shape[:-3]))

    status = elphmod.misc.StatusBar(sizes[comm.rank],
        title='calculate phonon spectral function')

    for my_w, w in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        G_inv = D[..., w] - (omega[w] + 1j * eta) ** 2 * np.eye(D.shape[-2])

        my_A[my_w] = 2 * omega[w] / np.pi * np.trace(np.linalg.inv(G_inv).imag,
            axis1=-2, axis2=-1)

        status.update()

    A = np.empty((len(omega), *D.shape[:-3]))

    comm.Allgatherv(my_A, (A, comm.allgather(my_A.size)))

    return A
