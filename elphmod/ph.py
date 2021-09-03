#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import numpy as np

from . import bravais, MPI
comm = MPI.comm

class Model(object):
    """Mass-spring model for the phonons.

    Parameters
    ----------
    flfrc : str
        File with interatomic force constants from ``q2r.x``.
    apply_asr : bool
        Apply acoustic sum rule correction to force constants?
    apply_asr_simple : bool
        Apply simple acoustic sum rule correction to force constants? This sets
        the self force constant to minus the sum of all other force constants.
    apply_rsr : bool
        Apply rotation sum rule correction to force constants?
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
    divide_mass : bool
        Divide force constants by atomic masses?
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
    atom_order : list of str
        Ordered list of atoms.
    data : ndarray
        Corresponding self and interatomic force constants.
    size : int
        Number of displacement directions/bands.
    nat : int
        Number of atoms.
    cells : list of tuple of int, optional
        Lattice vectors of unit cells if the model describes a supercell.
    """
    def D(self, q1=0, q2=0, q3=0):
        """Set up dynamical matrix for arbitrary q point."""

        q = np.array([q1, q2, q3])
        D = np.empty(self.data.shape, dtype=complex)

        for n in range(D.shape[0]):
            D[n] = self.data[n] * np.exp(-1j * np.dot(self.R[n], q))

            # Sign convention in do_q3r.f90 of QE:
            # 231  CALL cfft3d ( phid (:,j1,j2,na1,na2), &
            # 232       nr1,nr2,nr3, nr1,nr2,nr3, 1, 1 )
            # 233  phid(:,j1,j2,na1,na2) = &
            # 234       phid(:,j1,j2,na1,na2) / DBLE(nr1*nr2*nr3)
            # The last argument of cfft3d is the sign (+1).

        return D.sum(axis=0)

    def __init__(self, flfrc=None, apply_asr=False, apply_asr_simple=False,
        apply_rsr=False, phid=np.zeros((1, 1, 1, 1, 1, 3, 3)), amass=np.ones(1),
        at=np.eye(3), tau=np.zeros((1, 3)), atom_order=['X'], divide_mass=True,
        divide_ndegen=True):

        if comm.rank == 0:
            if flfrc is None:
                model = phid.copy(), amass, at, tau, atom_order
            else:
                model = read_flfrc(flfrc)

            # optionally, apply acoustic sum rule:

            if apply_asr_simple:
                asr(model[0])
        else:
            model = None

        model = comm.bcast(model)

        self.M, self.a, self.r, self.atom_order = model[1:]
        self.R, self.data = short_range_model(*model[:-1],
            divide_mass=divide_mass, divide_ndegen=divide_ndegen)
        self.size = self.data.shape[1]
        self.nat = self.size // 3

        if apply_asr or apply_rsr:
            sum_rule_correction(self, asr=apply_asr, rsr=apply_rsr,
                divide_mass=divide_mass)

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

        B1 = np.cross(N2, N3)
        B2 = np.cross(N3, N1)
        B3 = np.cross(N1, N2)

        ph = Model()
        ph.M = np.tile(self.M, N)
        ph.a = np.dot(np.array([N1, N2, N3]), self.a)
        ph.atom_order = list(self.atom_order) * N
        ph.size = self.size * N
        ph.nat = self.nat * N
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

                    const[R][A:A + self.size, B:B + self.size] = self.data[n]

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

    return [phid, amass[ityp], at, tau, atom_order]

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
                                const[R] = np.zeros((3 * nat, 3 * nat))

                            const[R][3 * na1:3 * na1 + 3,
                                     3 * na2:3 * na2 + 3] = C

    # convert dictionary into arrays:

    n = len(const)

    cells = np.array(list(const.keys()), dtype=np.int8)
    const = np.array(list(const.values()))

    # gather data of all processes:

    dims = np.array(comm.allgather(n))
    dim = dims.sum()

    allcells = np.empty((dim, 3), dtype=np.int8)
    allconst = np.empty((dim, 3 * nat, 3 * nat))

    comm.Allgatherv(cells[:n], (allcells, dims * 3))
    comm.Allgatherv(const[:n], (allconst, dims * (3 * nat) ** 2))

    # (see cdef _p_message message_vector in mpi4py/src/mpi4py/MPI/msgbuffer.pxi
    # for possible formats of second argument 'recvbuf')

    return allcells, allconst

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
        apply_asr=False, apply_asr_simple=False, apply_rsr=False):
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

    ph.R, ph.data = short_range_model(phid, ph.M, ph.a, ph.r)

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
