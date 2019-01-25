#/usr/bin/env python

import numpy as np

from . import bravais, dispersion, el, MPI, ph
comm = MPI.comm
info = MPI.info

def coupling(filename, nQ, nb, nk, bands, Q=None, nq=None, offset=0,
        completion=True, complete_k=False, squeeze=False, status=False,
        phase=False):
    """Read and complete electron-phonon matrix elements."""

    if Q is not None:
        nQ = len(Q)
    else:
        Q = np.arange(nQ, dtype=int) + 1

    sizes = MPI.distribute(nQ)

    dtype = complex if phase else float

    elph = np.empty((nQ, nb, bands, bands, nk, nk), dtype=dtype)

    my_elph = np.empty((sizes[comm.rank], nb, bands, bands, nk, nk),
        dtype=dtype)

    my_elph[:] = np.nan

    my_Q = np.empty(sizes[comm.rank], dtype=int)
    comm.Scatterv((Q, sizes), my_Q)

    for n, iq in enumerate(my_Q):
        if status:
            print("Read data for q point %d.." % iq)

        # TypeError: 'test' % 1
        # permitted: 'test' % np.array(1)

        with open(filename % iq) as data:
            for line in data:
                columns = line.split()

                if columns[0].startswith('#'):
                    continue

                k1, k2, k3, wk, ibnd, jbnd, nu \
                    = [int(i) - 1 for i in columns[:7]]

                indices = n, nu, ibnd - offset, jbnd - offset, k1, k2

                my_elph[indices] = float(columns[7])

                if phase:
                    my_elph[indices] += 1j * float(columns[8])

    if completion:
        for n, iq in enumerate(my_Q):
            if status:
                print("Complete data for q point %d.." % iq)

            for nu in range(nb):
                for ibnd in range(bands):
                    for jbnd in range(bands):
                        bravais.complete(my_elph[n, nu, ibnd, jbnd])

    if complete_k and nq: # to be improved considerably
        comm.Gatherv(my_elph, (elph, sizes * nb * bands * bands * nk * nk))

        elph_complete = np.empty((nq, nq, nb, bands, bands, nk, nk),
            dtype=dtype)

        if comm.rank == 0:
            symmetries_q = [image for name, image in bravais.symmetries(
                np.zeros((nq, nq)), unity=True)]

            symmetries_k = [image for name, image in bravais.symmetries(
                np.zeros((nk, nk)), unity=True)]

            done = set()
            q_irr = sorted(bravais.irreducibles(nq))

            for sym_q, sym_k in zip(symmetries_q, symmetries_k):
                for iq, (q1, q2) in enumerate(q_irr):
                    Q1, Q2 = sym_q[q1, q2]

                    if (Q1, Q2) in done:
                        continue

                    done.add((Q1, Q2))

                    for k1 in range(nk):
                        for k2 in range(nk):
                            K1, K2 = sym_k[k1, k2]

                            elph_complete[Q1, Q2, ..., K1, K2] \
                                = elph[iq, ..., k1, k2]

        comm.Bcast(elph_complete)
        elph = elph_complete
    else:
        comm.Allgatherv(my_elph, (elph, sizes * nb * bands * bands * nk * nk))

    return elph[..., 0, 0, :, :] if bands == 1 and squeeze else elph

def read_EPW_output(epw_out, q, nq, nb, nk, bands=1,
                    eps=1e-4, squeeze=False, status=False, epf=False):
    """Read electron-phonon coupling from EPW output file."""

    elph = np.empty((len(q), nb, bands, bands, nk, nk),
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

                    for _ in range(bands * bands * nb):
                        columns = next(data).split()

                        ibnd, jbnd, nu = [int(i) - 1 for i in columns[:3]]

                        if epf:
                            elph[iq, nu, ibnd, jbnd, k1, k2] = complex(
                                float(columns[-2]), float(columns[-1]))
                        else:
                            elph[iq, nu, ibnd, jbnd, k1, k2] = float(
                                columns[-1])

        if np.isnan(elph).any():
            print("Warning: EPW output incomplete!")

        if epf:
            elph *= 1e-3 ** 1.5 # meV^(3/2) to eV^(3/2)
        else:
            elph *= 1e-3 # meV to eV

    comm.Bcast(elph)

    return elph[:, :, 0, 0, :, :] if bands == 1 and squeeze else elph

def read_xml_files(filename, q, rep, bands, nbands, nk, squeeze=True, status=True,
        angle=120, angle0=0):
    """Read XML files with coupling in displacement basis from QE (nosym)."""

    if not hasattr(q, '__len__'):
        q = range(q)

    if not hasattr(rep, '__len__'):
        rep = range(rep)

    if not hasattr(bands, '__len__'):
        bands = [bands]

    t1, t2 = bravais.translations(angle, angle0)

    sizes = MPI.distribute(len(q))

    elph = np.empty((len(q), len(rep), len(bands), len(bands), nk, nk),
        dtype=complex)

    my_elph = np.empty((sizes[comm.rank],
        len(rep), len(bands), len(bands), nk, nk), dtype=complex)

    band_select = np.empty(nbands, dtype=int)
    band_select[:] = -1

    for n, m in enumerate(bands):
        band_select[m] = n

    my_q = np.empty(sizes[comm.rank], dtype=int)
    comm.Scatterv((np.array(q), sizes), my_q)

    for my_iq, iq in enumerate(my_q):
        if status:
            print("Read data for q point %d.." % (iq + 1))

        for my_irep, irep in enumerate(rep):
            with open(filename % (iq + 1, irep + 1)) as data:
                def goto(pattern):
                    for line in data:
                        if pattern in line:
                            return line

                goto("<NUMBER_OF_K ")
                if nk != int(np.sqrt(int(next(data)))):
                    print("Wrong number of k points!")

                goto("<NUMBER_OF_BANDS ")
                if nbands != int(next(data)):
                    print("Wrong number of bands!")

                for ik in range(nk * nk):
                    goto("<COORDINATES_XK ")
                    k = list(map(float, next(data).split()))[:2]

                    k1 = int(round(np.dot(k, t1) * nk)) % nk
                    k2 = int(round(np.dot(k, t2) * nk)) % nk

                    goto("<PARTIAL_ELPH ")

                    for n in band_select:
                        for m in band_select:
                            if n < 0 or m < 0:
                                next(data)
                            else:
                                my_elph[my_iq, my_irep, n, m, k1, k2] = complex(
                                    *list(map(float, next(data).split(","))))

    comm.Allgatherv(my_elph, (elph,
        sizes * len(rep) * len(bands) * len(bands) * nk * nk))

    return elph[..., 0, 0, :, :] if len(bands) == 1 and squeeze else elph

def write_xml_files(filename, data, angle=120, angle0=0):
    """Write XML files with coupling in displacement basis."""

    if comm.rank != 0:
        return

    if data.ndim == 4:
        data = data[:, :, np.newaxis, np.newaxis, :, :]

    nQ, nb, nbands, nbands, nk, nk = data.shape

    t1, t2 = bravais.translations(angle, angle0)
    u1, u2 = bravais.reciprocals(t1, t2)

    for iq in range(nQ):
        for irep in range(nb):
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

                        k = (k1 * u1 + k2 * u2) / nk

                        xml.write("""
    <K_POINT.%d>
      <COORDINATES_XK type="real" size="3" columns="3">
%23.15E %23.15E %23.15E
      </COORDINATES_XK>
      <PARTIAL_ELPH type="complex" size="%d">"""
                            % (ik, k[0], k[1], 0.0, nbands * nbands))

                        for i in range(nbands):
                            for j in range(nbands):
                                g = data[iq, irep, i, j, k1, k2]

                                xml.write("""
%23.15E,%23.15E""" % (g.real, g.imag))

                        xml.write("""
      </PARTIAL_ELPH>
    </K_POINT.%d>""" % ik)

                xml.write("""
  </PARTIAL_EL_PHON>
</Root>
""")

def read(filename, nq, bands):
    """Read and complete Fermi-surface averaged electron-phonon coupling."""

    elph = np.empty((nq, nq, bands))

    with open(filename) as data:
        for line in data:
            columns = line.split()

            q1 = int(columns[0])
            q2 = int(columns[1])

            for Q1, Q2 in bravais.images(q1, q2, nq):
                for band in range(bands):
                    elph[Q1, Q2, band] = float(columns[2 + band])

    return elph

def epw(epmatwp, wigner, outdir, nbndsub, nmodes, nk, nq, q='wedge',
        orbital_basis=False, wannier=None, displacement_basis=True, ifc=None,
        read_eigenvectors=False):
    """Simulate second part of EPW: coarse Wannier to fine Bloch basis.

    The purpose of this routine is full control of the coupling's complex phase.

    Parameters
    ----------
    epmatwp : str
        File with electron-phonon coupling in Wannier basis produced by EPW.
    wigner : str
        File with lattice vectors in Wigner-Seitz cell belonging to 'epmatwp'.

        This file is not produced by EPW by default. It contains the variables

            'nrr_k', 'irvec_k', 'ndegen_k', 'wslen_k',
            'nrr_q', 'irvec_q', 'ndegen_q', 'wslen_q',
            'nrr_g', 'irvec_g', 'ndegen_g', 'wslen_g',

        which are allocated and calculated in the EPW source file 'wigner.f90',
        in the given order and in binary representation without any separators.

    outdir : str
        Directory where (some of) the following output files are stored:

            el-ph-(q point).dat        (to be read by `bravais.coupling`)
            electron_eigenvectors.dat
            electron_eigenvalues.dat
            phonon_eigenvectors.dat    (divided by square roots of masses)
            phonon_eigenvalues.dat     (phonon frequencies  s q u a r e d)

    nbndsub : int
        Number of electron bands or Wannier functions.
    nmodes : int
        Number of phonon modes (three times the number of atoms).
    nk : int
        Number of k points per dimension.
    nq : int
        Number of q points per dimension.
    q : str or list of 2-tuples, optional
        Requested q points. The possible values are:

            'wedge': Irreducible wedge of the uniform `nq` x `nq` mesh.
                     This should be consistent with Quantum ESPRESSO.

             'mesh': Full `nq` x `nq` mesh.

                 or: Custom q points in crystal coordinates q1, q2 in [0, 2pi).

    orbital_basis : bool, optional
        Stay in the orbital basis or transform to band basis?
    wannier : str
        File with Wannier Hamiltonian.
    displacement_basis : bool, optional
        Stay in the displacement basis or transform to mode basis?
    ifc : str, optional
        File with interatomic force constants.
    read_eigenvectors : bool, optional
        Read electron and phonon eigenvectors from previously written files
        instead of calculating them? This option can be used to guarantee the
        same gauge in different calculations, especially if the implementation
        of NumPy's diagonalization routines is not deterministic.
    """
    nat = nmodes // 3

    if q == 'wedge':
        # generate same list of irreducible q points as Quantum ESPRESSO:

        q_int = sorted(bravais.irreducibles(nq))
        q = np.array(q_int, dtype=float) / nq * 2 * np.pi

    elif q == 'mesh':
        q_int = [(q1, q2) for q1 in range(nq) for q2 in range(nq)]
        q = np.array(q_int, dtype=float) / nq * 2 * np.pi

    else:
        q = np.array(q) % (2 * np.pi)
        q_int = np.round(q * nq / (2 * np.pi)).astype(int)

    # read lattice vectors within Wigner-Seitz cell:

    (nrr_k, irvec_k, ndegen_k, wslen_k,
     nrr_q, irvec_q, ndegen_q, wslen_q,
     nrr_g, irvec_g, ndegen_g, wslen_g) = bravais.read_wigner_file(wigner, nat)

    # read coupling in Wannier basis from EPW output:
    # ('epmatwp' allocated and printed in 'ephwann_shuffle.f90')

    shape = (nrr_g, nmodes, nrr_k, nbndsub, nbndsub)

    if comm.rank == 0:
        with open(epmatwp) as data:
            g = np.fromfile(data, dtype=np.complex128)

        g = np.reshape(g, shape)

    else:
        g = np.empty(shape, dtype=np.complex128)

    comm.Bcast(g)

    # transfrom from Wannier to Bloch basis:
    # (see 'wan2bloch.f90' in EPW 5.0)

    # electrons 1: transform from real to k space:
    #
    # from g(nrr_g, nmodes, nrr_k,  nbndsub, nbndsub)
    # to   g(nrr_g, nmodes, nk, nk, nbndsub, nbndsub)

    info("Real space to k..")

    sizes, bounds = MPI.distribute(nk * nk, bounds=True)

    my_g = np.empty((sizes[comm.rank], nrr_g, nmodes, nbndsub, nbndsub),
        dtype=np.complex128)

    tmp = np.empty_like(g)

    for my_ik, ik in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        k1 = ik // nk
        k2 = ik % nk

        print('k = (%d, %d)' % (k1, k2))

        k = np.array([k1, k2], dtype=float) / nk * 2 * np.pi

        for irk in range(nrr_k):
            tmp[:, :, irk] = g[:, :, irk] \
                * np.exp(1j * np.dot(k, irvec_k[irk])) / ndegen_k[irk]

        my_g[my_ik] = tmp.sum(axis=2)

    g = np.empty((nk, nk, nrr_g, nmodes, nbndsub, nbndsub), dtype=np.complex128)

    comm.Allgatherv(my_g, (g, sizes * nrr_g * nmodes * nbndsub * nbndsub))

    g = np.transpose(g, axes=(2, 3, 0, 1, 4, 5))

    # phonons 1: transform from real to q space:
    #
    # from g(nrr_g , nmodes, nk, nk, nbndsub, nbndsub)
    # to   g(len(q), nmodes, nk, nk, nbndsub, nbndsub)

    info("Real space to q..")

    sizes, bounds = MPI.distribute(len(q), bounds=True)

    my_g = np.empty((sizes[comm.rank], nmodes, nk, nk, nbndsub, nbndsub),
        dtype=np.complex128)

    tmp = np.empty_like(g)
    exp = np.empty(nrr_g, dtype=complex)

    block = [slice(3 * na, 3 * (na + 1)) for na in range(nat)]

    for my_iq, iq in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
        print('q = %d' % iq)

        for irg in range(nrr_g):
            exp[irg] = np.exp(1j * np.dot(q[iq], irvec_g[irg]))

            for na in range(nat):
                if ndegen_g[na, irg]:
                    tmp[irg, block[na]] = g[irg, block[na]] \
                        * exp[irg] / ndegen_g[na, irg]
                else:
                    tmp[irg, block[na]] = 0

        my_g[my_iq] = tmp.sum(axis=0)

    g = np.empty((len(q), nmodes, nk, nk, nbndsub, nbndsub),
        dtype=np.complex128)

    comm.Allgatherv(my_g, (g, sizes * nmodes * nk * nk * nbndsub * nbndsub))

    if not orbital_basis:
        # electrons 2: transform from orbital to band basis:
        # (the meaning of the last two indices changes)

        info('Orbital to band..')

        my_g = np.empty((sizes[comm.rank], nmodes, nk, nk, nbndsub, nbndsub),
            dtype=np.complex128)

        if read_eigenvectors:
            U = np.empty((nk, nk, nbndsub, nbndsub), dtype=complex)

            if comm.rank == 0:
                read_electron_eigenvectors(
                    '%s/electron_eigenvectors.dat' % outdir, U)

            comm.Bcast(U)
        else:
            H = el.hamiltonian(wannier)
            e, U = dispersion.dispersion_full_nosym(H, nk, vectors=True)

        for my_iq, iq in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
            q1, q2 = q_int[iq]

            q1 *= nk // nq
            q2 *= nk // nq

            for k1 in range(nk):
                kq1 = (k1 + q1) % nk

                for k2 in range(nk):
                    kq2 = (k2 + q2) % nk

                    for i in range(nmodes):
                        my_g[my_iq, i, k1, k2] = U[k1, k2].T.dot(
                           g[   iq, i, k1, k2]).dot(U[kq1, kq2].conj())

        g = np.empty((len(q), nmodes, nk, nk, nbndsub, nbndsub),
            dtype=np.complex128)

        comm.Allgatherv(my_g, (g, sizes * nmodes * nk * nk * nbndsub * nbndsub))

    if not displacement_basis:
        # phonons 2: transform from displacement to mode basis:
        # (the meaning of the second index changes)

        info('Displacement to mode..')

        if read_eigenvectors:
            u = np.empty((len(q), nmodes, nmodes), dtype=complex)

            if comm.rank == 0:
                read_phonon_eigenvectors(
                    '%s/phonon_eigenvectors.dat' % outdir, u)

            comm.Bcast(u)
        else:
            phid, amass, at, tau = ph.model(ifc, apply_asr=True)
            D = ph.dynamical_matrix(phid, amass, at, tau)

            w2, u = dispersion.dispersion(D, q, vectors=True)

            for na in range(nat):
                u[:, 3 * na:3 * na + 3] /= np.sqrt(amass[na])

        my_g = np.empty((sizes[comm.rank], nmodes, nk, nk, nbndsub, nbndsub),
            dtype=np.complex128)

        for my_iq, iq in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
            my_g[my_iq] = np.einsum('iklnm,ij->jklnm', g[iq], u[iq])

        comm.Allgatherv(my_g, (g, sizes * nmodes * nk * nk * nbndsub * nbndsub))

    # Write results to disk:

    if comm.rank != 0:
        return

    write_coupling('%s/el-ph-%%d.dat' % outdir, g,
        orbital_basis, displacement_basis)

    if not orbital_basis and not read_eigenvectors:
        write_electron_eigenvectors('%s/electron_eigenvectors.dat' % outdir, U)
        write_electron_eigenvalues('%s/electron_eigenvalues.dat' % outdir, e)

    if not displacement_basis and not read_eigenvectors:
        write_phonon_eigenvectors('%s/phonon_eigenvectors.dat' % outdir, u)
        write_phonon_eigenvalues('%s/phonon_eigenvalues.dat' % outdir, w2)

def write_coupling(filename, g, orbital_basis=False, displacement_basis=False):
    """Write electron-phonon coupling to text file.

    Parameters
    ----------
    filename : str
        Filename with placeholder "%d" for q-point index.
    g : ndarray
        Electron-phonon coupling.
    orbital_basis : bool, optional
        Coupling given in orbital instead of electron band basis?
    displacement_basis : bool, optional
        Coupling given in displacement instead of phonon band basis?

    See Also
    --------
    coupling
    """
    nQ, nmodes, nk, nk, nbndsub, nbndsub = g.shape

    for iq in range(nQ):
        with open(filename % (iq + 1), 'w') as data:
            data.write("""#
#  Electron-phonon matrix elements
#
#    k1,2,3: k-point indices
#    w:      k-point weight
#    n:      1st electron {0} index
#    m:      2nd electron {0} index
#    i:      {1} index
#    ElPh:   <k+q m| dV/du(q, i) |k n>
#
#k1 k2 k3  w  n  m  i        Re[ElPh]        Im[ElPh]
#----------------------------------------------------""".format(
        'orbital'             if      orbital_basis else 'band',
        'atomic displacement' if displacement_basis else 'phonon mode'))

            for k1 in range(nk):
                for k2 in range(nk):
                    for n in range(nbndsub):
                        for m in range(nbndsub):
                            for i in range(nmodes):
                                data.write("""
%3d%3d%3d%3d%3d%3d%3d%16.8E%16.8E""" % (k1 + 1, k2 + 1, 1, 1,
                                        n + 1, m + 1, i + 1,
                                        g[iq, i, k1, k2, n, m].real,
                                        g[iq, i, k1, k2, n, m].imag))

def write_electron_eigenvectors(filename, U):
    """Write eigenvectors of Wannier Hamiltonian to text file.

    Parameters
    ----------
    filename : str
        Name of text file.
    U : ndarray
        Eigenvectors on uniform k mesh.
    """
    nk, nk, nbndsub, nbnsub = U.shape

    with open(filename, 'w') as data:
        data.write("""#
#  Eigenvectors of Wannier Hamiltonian
#
#    k1,2,3: k-point indices
#    a:      orbital index
#    n:      band index
#    U:      <k a|k n>
#
#k1 k2 k3  a  n           Re[U]           Im[U]
#----------------------------------------------""")

        for k1 in range(nk):
            for k2 in range(nk):
                for a in range(nbndsub):
                    for n in range(nbndsub):
                        data.write("""
%3d%3d%3d%3d%3d%16.8E%16.8E""" % (k1 + 1, k2 + 1, 1, a + 1, n + 1,
                                  U[k1, k2, a, n].real,
                                  U[k1, k2, a, n].imag))

def read_electron_eigenvectors(filename, U):
    """Read eigenvectors of Wannier Hamiltonian from text file.

    See Also
    --------
    write_electron_eigenvectors
    """
    with open(filename) as data:
        for line in data:
            if not line.startswith('#'):
                columns = line.split()

                k1, k2, _, a, n = [-1 + int(x) for x in columns[:5]]
                Re, Im          = [   float(x) for x in columns[5:]]

                U[k1, k2, a, n] = Re + 1j * Im

def write_electron_eigenvalues(filename, e):
    """Write eigenvalues of Wannier Hamiltonian to text file.

    Parameters
    ----------
    filename : str
        Name of text file.
    e : ndarray
        Eigenvalues on uniform k mesh.
    """
    nk, nk, nbndsub = e.shape

    with open('%s/electron_eigenvalues.dat' % outdir, 'w') as data:
        data.write("""#
#  Eigenvalues of Wannier Hamiltonian
#
#    k1,2,3: k-point indices
#    n:      band index
#    eps:    <k n|H|k n>
#
#k1 k2 k3  n             eps
#---------------------------""")

        for k1 in range(nk):
            for k2 in range(nk):
                for n in range(nbndsub):
                    data.write("""
%3d%3d%3d%3d%16.8E""" % (k1 + 1, k2 + 1, 1, n + 1, e[k1, k2, n]))

def write_phonon_eigenvectors(filename, u):
    """Write eigenvectors of dynamical matrix to text file.

    Parameters
    ----------
    filename : str
        Name of text file.
    u : ndarray
        Eigenvectors for selected q points.
    """
    nQ, nmodes, nmodes = u.shape

    with open('%s/phonon_eigenvectors.dat' % outdir, 'w') as data:
        data.write("""#
#  Eigenvectors of dynamical matrix
#
#    q:  q-point index
#    x:  atomic displacement index
#    nu: phonon mode index
#    u:  <q x|k nu>
#
# q  x nu           Re[u]           Im[u]
#----------------------------------------""")

        for iq in range(nQ):
            for x in range(nmodes):
                for nu in range(nmodes):
                    data.write("""
%3d%3d%3d%16.8E%16.8E""" % (iq + 1, x + 1, nu + 1,
                            u[iq, x, nu].real,
                            u[iq, x, nu].imag))

def read_phonon_eigevectors(filename, u):
    """Read eigenvectors of dynamical matrix from text file.

    See Also
    --------
    write_phonon_eigenvectors
    """
    with open('%s/phonon_eigenvectors.dat' % outdir) as data:
        for line in data:
            if not line.startswith('#'):
                columns = line.split()

                iq, x, nu = [-1 + int(x) for x in columns[:3]]
                Re, Im    = [   float(x) for x in columns[3:]]

                u[iq, x, nu] = Re + 1j * Im

def write_phonon_eigenvalues(filename, w2):
    """Write eigenvalues of dynamical matrix to text file.

    Parameters
    ----------
    filename : str
        Name of text file.
    w2 : ndarray
        Eigenvalues for selected q points.
    """
    nQ, nmodes = w2.shape

    with open('%s/phonon_eigenvalues.dat' % outdir, 'w') as data:
        data.write("""#
#  Eigenvalues of dynamical matrix
#
#    q:  q-point index
#    nu: phonon mode index
#    w2: <q nu|D|q nu>
#
# q nu              w2
#---------------------""")

        for iq in range(nQ):
            for nu in range(nmodes):
                data.write("""
%3d%3d%16.8E""" % (iq + 1, nu + 1, w2[iq, nu]))
