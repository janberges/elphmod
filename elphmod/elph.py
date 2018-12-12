#/usr/bin/env python

import numpy as np

from . import bravais, dispersion, el, MPI
comm = MPI.comm

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

def epw(epmatwp, wigner, wannier, outdir, nbndsub, nmodes, nk, nq, n, mu=0.0):
    """Simulate second part of EPW: coarse Wannier to fine Bloch basis.

    Only the transformation from the displacement to the mode basis is omitted.
    The purpose of this routine is full control of the coupling's complex phase.

    Parameters
    ----------
    epmatwp : string
        File with electron-phonon coupling in Wannier basis produced by EPW.
    wigner : string
        File with lattice vectors in Wigner-Seitz cell belonging to 'epmatwp'.

        This file is not produced by EPW by default. It contains the variables

            'nrr_k', 'irvec_k', 'ndegen_k', 'wslen_k',
            'nrr_q', 'irvec_q', 'ndegen_q', 'wslen_q',
            'nrr_g', 'irvec_g', 'ndegen_g', 'wslen_g',

        which are allocated and calculated in the EPW source file 'wigner.f90',
        in the given order and in binary representation without any separators.

    wannier : File with Wannier Hamiltonian.
    outdir : string
        Directory where the following output files are stored:

            FILENAME             CONTENT                    TO BE READ BY
            el-ph-(q point).dat  electron-phonon coupling   bravais.coupling
            eigenvectors.dat     "orbital band characters"  -
            eigenvalues.dat      electron energies          -

    nbndsub : int
        Number of electron bands or Wannier functions.
    nmodes : int
        Number of phonon modes (three times the number of atoms).
    nk : int
        Number of k points per dimension.
    nq : int
        Number of q points per dimension.
    n : int
        Index of electron band for which to calculate results.
    mu : float, optional
        Fermi level to be subtracted from electron energies before saving.
    """
    nat = nmodes // 3

    # generate same list of irreducible q points as Quantum ESPRESSO:

    q_int = sorted(bravais.irreducibles(nq))
    q = np.array(q_int, dtype=float) / nq * 2 * np.pi

    # read lattice vectors within Wigner-Seitz cell:

    with open(wigner, 'rb') as data:
        integer = np.int32
        double  = np.float64

        nrr_k    = np.fromfile(data, integer, 1)[0]
        irvec_k  = np.fromfile(data, integer, nrr_k * 3)
        irvec_k  = irvec_k.reshape((nrr_k, 3))[:, :2]
        ndegen_k = np.fromfile(data, integer, nrr_k)
        wslen_k  = np.fromfile(data, double, nrr_k)

        nrr_q    = np.fromfile(data, integer, 1)[0]
        irvec_q  = np.fromfile(data, integer, nrr_q * 3)
        irvec_q  = irvec_q.reshape((nrr_q, 3))[:, :2]
        ndegen_q = np.fromfile(data, integer, nat * nat * nrr_q)
        ndegen_q = ndegen_q.reshape((nat, nat, nrr_q))
        wslen_q  = np.fromfile(data, double, nrr_q)

        nrr_g    = np.fromfile(data, integer, 1)[0]
        irvec_g  = np.fromfile(data, integer, nrr_g * 3)
        irvec_g  = irvec_g.reshape((nrr_g, 3))[:, :2]
        ndegen_g = np.fromfile(data, integer, nat * nrr_g)
        ndegen_g = ndegen_g.reshape((nat, nrr_g))
        wslen_g  = np.fromfile(data, double, nrr_g)

    # read coupling in Wannier basis from EPW output:
    # ('epmatwp' allocated and printed in 'ephwann_shuffle.f90')

    with open(epmatwp) as data:
        g = np.fromfile(data, dtype=np.complex128)

    g = np.reshape(g, (nrr_g, nmodes, nrr_k, nbndsub, nbndsub))

    # transfrom from Wannier to Bloch basis:
    # (see 'wan2bloch.f90' in EPW 5.0)

    # electrons 1: transform from real to k space:
    #
    # from g(nrr_g, nmodes, nrr_k,  nbndsub, nbndsub)
    # to   g(nrr_g, nmodes, nk, nk, nbndsub, nbndsub)


    g_new = np.empty((nrr_g, nmodes, nk, nk, nbndsub, nbndsub), dtype=complex)

    tmp = np.empty_like(g)

    for k1 in range(nk):
        for k2 in range(nk):
            print('k = (%d, %d)' % (k1, k2))

            k = np.array([k1, k2], dtype=float) / nk * 2 * np.pi

            for irk in range(nrr_k):
                tmp[:, :, irk] = g[:, :, irk] \
                    * np.exp(1j * np.dot(k, irvec_k[irk])) / ndegen_k[irk]

            g_new[:, :, k1, k2] = tmp.sum(axis=2)

    g = g_new

    # phonons 1: transform from real to q space:
    #
    # from g(nrr_g , nmodes, nk, nk, nbndsub, nbndsub)
    # to   g(len(q), nmodes, nk, nk, nbndsub, nbndsub)

    g_new = np.empty((len(q), nmodes, nk, nk, nbndsub, nbndsub), dtype=complex)

    tmp = np.empty_like(g)
    exp = np.empty(nrr_g, dtype=complex)

    block = [slice(3 * na, 3 * (na + 1)) for na in range(nat)]

    for iq in range(len(q)):
        print('q = %d' % iq)

        for irg in range(nrr_g):
            exp[irg] = np.exp(1j * np.dot(q[iq], irvec_g[irg]))

            for na in range(nat):
                if ndegen_g[na, irg]:
                    tmp[irg, block[na]] = g[irg, block[na]] \
                        * exp[irg] / ndegen_g[na, irg]
                else:
                    tmp[irg, block[na]] = 0

        g_new[iq] = tmp.sum(axis=0)

    g = g_new

    # electrons 2: transform from orbital to band basis:
    #
    # from g(len(q), nmodes, nk, nk, nbndsub, nbndsub)
    # to   g(len(q), nmodes, nk, nk) (only to one band currently)

    print('Orbital to band..')

    g_new = np.empty((len(q), nmodes, nk, nk), dtype=complex)

    H = el.hamiltonian(wannier)
    e, U = dispersion.dispersion_full_nosym(H, nk, vectors=True)

    for iq in range(len(q)):
        q1, q2 = q_int[iq]

        q1 *= nk // nq
        q2 *= nk // nq

        for k1 in range(nk):
            kq1 = (k1 + q1) % nk

            for k2 in range(nk):
                kq2 = (k2 + q2) % nk

                for i in range(nmodes):
                    g_new[iq, i, k1, k2] = U[k1, k2, :, n].dot(
                        g[iq, i, k1, k2]).dot(U[kq1, kq2, :, n].conj())

    g = g_new

    # Write results to disk:

    for iq in range(len(q)):
        with open('%s/el-ph-%d.dat' % (outdir, iq + 1), 'w') as data:
            data.write("""#
    #  Electron-phonon matrix elements
    #
    #    k1,2,3: k-point indices
    #    w:      k-point weight
    #    n:      1st electronic band index
    #    m:      2nd electronic band index
    #    i:      atomic displacement index
    #    ElPh:   <k+q m| dV/du(q, i) |k n>
    #
    #k1 k2 k3  w  n  m  i        Re[ElPh]        Im[ElPh]
    #----------------------------------------------------""")

            for k1 in range(nk):
                for k2 in range(nk):
                    for i in range(nmodes):
                        data.write("""
    %3d%3d%3d%3d%3d%3d%3d%16.8E%16.8E""" % (k1 + 1, k2 + 1, 1, 1, 1, 1, i + 1,
                            g[iq, i, k1, k2].real, g[iq, i, k1, k2].imag))

    with open('%s/eigenvectors.dat' % outdir, 'w') as data:
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
                    data.write("""
    %3d%3d%3d%3d%3d%16.8E%16.8E""" % (k1 + 1, k2 + 1, 1, a + 1, 1,
                        U[k1, k2, a, n].real, U[k1, k2, a, n].imag))

    e -= mu

    with open('%s/eigenvalues.dat' % outdir, 'w') as data:
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
                data.write("""
    %3d%3d%3d%3d%16.8E""" % (k1 + 1, k2 + 1, 1, 1, e[k1, k2, n]))
