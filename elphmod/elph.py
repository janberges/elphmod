#/usr/bin/env python

import numpy as np

from . import bravais, dispersion, el, misc, MPI, ph
comm = MPI.comm

class Model(object):
    """Localized model for electron-phonon coupling.

    The methods of this class follow 'wan2bloch.f90' of EPW 5.0.
    """
    def g(self, q1=0, q2=0, q3=0, k1=0, k2=0, k3=0, broadcast=True, comm=comm):
        nRq, nph, nRk, nel, nel = self.data.shape

        q = np.array([q1, q2, q3])
        k = np.array([k1, k2, k3])

        if self.q is None or np.any(q != self.q):
            self.q = q

            sizes, bounds = MPI.distribute(nRq, bounds=True, comm=comm)

            my_g = np.empty((sizes[comm.rank], nph, nRk, nel, nel),
                dtype=complex)

            for my_n, n in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
                my_g[my_n] = self.data[n] * np.exp(1j * np.dot(self.Rg[n], q))

                # Sign convention in bloch2wan.f90 of EPW:
                # 1273  cfac = exp( -ci*rdotk ) / dble(nq)
                # 1274  epmatwp(:,:,:,:,ir) = epmatwp(:,:,:,:,ir)
                #           + cfac * epmatwe(:,:,:,:,iq)

            self.gq = np.empty((nph, nRk, nel, nel), dtype=complex)

            comm.Allreduce(my_g.sum(axis=0), self.gq)

        sizes, bounds = MPI.distribute(nRk, bounds=True, comm=comm)

        my_g = np.empty((sizes[comm.rank], nph, nel, nel), dtype=complex)

        for my_n, n in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
            my_g[my_n] = self.gq[:, n] * np.exp(1j * np.dot(self.Rk[n], k))

            # Sign convention in bloch2wan.f90 of EPW:
            # 1148  cfac = exp( -ci*rdotk ) / dble(nkstot)
            # 1149  epmatw( :, :, ir) = epmatw( :, :, ir)
            #           + cfac * epmats( :, :, ik)

        if broadcast or comm.rank == 0:
            g = np.empty((nph, nel, nel), dtype=complex)
        else:
            g = None

        comm.Reduce(my_g.sum(axis=0), g)

        if broadcast:
            comm.Bcast(g)

        return g

    def __init__(self, epmatwp, wigner, el, ph):
        self.el = el
        self.ph = ph

        # read lattice vectors within Wigner-Seitz cell:

        (nrr_k, irvec_k, ndegen_k, wslen_k,
         nrr_q, irvec_q, ndegen_q, wslen_q,
         nrr_g, irvec_g, ndegen_g, wslen_g) = bravais.read_wigner_file(wigner,
            nat=ph.nat)

        self.Rk = irvec_k
        self.Rg = irvec_g

        # read coupling in Wannier basis from EPW output:
        # ('epmatwp' allocated and printed in 'ephwann_shuffle.f90')

        shape = (nrr_g, ph.size, nrr_k, el.size, el.size)

        if comm.rank == 0:
            with open(epmatwp) as data:
                g = np.fromfile(data, dtype=np.complex128)

            g = np.reshape(g, shape)

            # undo supercell double counting:

            for irk in range(nrr_k):
                g[:, :, irk] /= ndegen_k[irk]

            block = [slice(3 * na, 3 * (na + 1)) for na in range(ph.nat)]

            for irg in range(nrr_g):
                for na in range(ph.nat):
                    if ndegen_g[na, irg]:
                        g[irg, block[na]] /= ndegen_g[na, irg]
                    else:
                        g[irg, block[na]] = 0.0

            # divide by square root of atomic masses:

            for na in range(ph.nat):
                g[:, block[na]] /= np.sqrt(ph.M[na])

        else:
            g = np.empty(shape, dtype=np.complex128)

        comm.Bcast(g)

        self.data = g

        self.q = None

    def sample(self, q, nk, U=None, u=None,
            broadcast=True, shared_memory=False):
        """Sample coupling for given q and k points and transform to band basis.

        One purpose of this routine is full control of the complex phase.

        Parameters
        ----------
        q : list of 2-tuples
            q points in crystal coordinates q1, q2 in [0, 2pi).
        nk : int
            Number of k points per dimension.
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

        scale = 2 * np.pi / nk

        nel = self.el.size if U is None else U.shape[-1]
        nph = self.ph.size if u is None else u.shape[-1]

        my_g = np.empty((sizes[row.rank], nph, nk, nk, nel, nel), dtype=complex)

        status = misc.StatusBar(sizes[comm.rank] * nk * nk,
            title='sample coupling')

        for my_iq, iq in enumerate(range(*bounds[row.rank:row.rank + 2])):
            q1, q2 = q[iq]

            Q1 = int(round(q1 / scale))
            Q2 = int(round(q2 / scale))

            for K1 in range(nk):
                KQ1 = (K1 + Q1) % nk
                k1 = K1 * scale

                for K2 in range(nk):
                    KQ2 = (K2 + Q2) % nk
                    k2 = K2 * scale

                    gqk = self.g(q1=q1, q2=q2, k1=k1, k2=k2,
                        broadcast=False, comm=col)

                    if col.rank == 0:
                        if U is not None:
                            gqk = np.einsum('xab,an,bm->xnm',
                                gqk, U[K1, K2], U[KQ1, KQ2].conj())

                        if u is not None:
                            gqk = np.einsum('xab,xu->uab', gqk, u[iq])

                        my_g[my_iq, :, K1, K2, :, :] = gqk

                    status.update()

        node, images, g = MPI.shared_array((len(q), nph, nk, nk, nel, nel),
            dtype=complex,
            shared_memory=shared_memory,
            single_memory=not broadcast)

        if col.rank == 0:
            row.Gatherv(my_g, (g, row.gather(my_g.size)))

        col.Barrier() # should not be necessary

        if node.rank == 0:
            images.Bcast(g)

        node.Barrier()

        return g

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

    elph = np.empty((nQ, nb, nk, nk, bands, bands), dtype=dtype)

    my_elph = np.empty((sizes[comm.rank], nb, nk, nk, bands, bands),
        dtype=dtype)

    my_elph[:] = np.nan

    my_Q = np.empty(sizes[comm.rank], dtype=int)
    comm.Scatterv((Q, sizes), my_Q)

    band_slice = slice(-5, -2) if phase else slice(-4, -1)

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

                k1, k2         = [int(i) - 1 for i in columns[:2]]
                ibnd, jbnd, nu = [int(i) - 1 for i in columns[band_slice]]

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
                print("Complete data for q point %d.." % iq)

            for nu in range(nb):
                for ibnd in range(bands):
                    for jbnd in range(bands):
                        bravais.complete(my_elph[n, nu, :, :, ibnd, jbnd])

    if complete_k and nq: # to be improved considerably
        comm.Gatherv(my_elph, (elph, sizes * nb * nk * nk * bands * bands))

        elph_complete = np.empty((nq, nq, nb, nk, nk, bands, bands),
            dtype=dtype)

        if comm.rank == 0:
            for nu in range(nb):
                for ibnd in range(bands):
                    for jbnd in range(bands):
                        elph_complete[:, :, nu, :, :, ibnd, jbnd] = \
                            bravais.complete_k(elph[:, nu, :, :, ibnd, jbnd], nq)

        comm.Bcast(elph_complete)
        elph = elph_complete
    else:
        comm.Allgatherv(my_elph, (elph, sizes * nb * nk * nk * bands * bands))

    return elph[..., 0, 0] if bands == 1 and squeeze else elph

def read_EPW_output(epw_out, q, nq, nb, nk, bands=1,
                    eps=1e-4, squeeze=False, status=False, epf=False):
    """Read electron-phonon coupling from EPW output file."""

    elph = np.empty((len(q), nb, nk, nk, bands, bands),
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
                            elph[iq, nu, k1, k2, ibnd, jbnd] = complex(
                                float(columns[-2]), float(columns[-1]))
                        else:
                            elph[iq, nu, k1, k2, ibnd, jbnd] = float(
                                columns[-1])

        if np.isnan(elph).any():
            print("Warning: EPW output incomplete!")

        if epf:
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
            print("Read displacement pattern for q point %d.." % (iq + 1))

        with open(filename % (iq + 1)) as data:
            def goto(pattern):
                for line in data:
                    if pattern in line:
                        return line

            goto("<NUMBER_IRR_REP ")
            if nrep != int(next(data)):
                print("Wrong number of representations!")

            for irep in range(nrep):
                goto("<DISPLACEMENT_PATTERN ")

                for jrep in range(nrep):
                    my_patterns[my_iq, irep, jrep] = float(
                        next(data).split(",")[0])

    comm.Allgatherv(my_patterns, (patterns, sizes * nrep * nrep))

    return patterns

def read_xml_files(filename, q, rep, bands, nbands, nk, squeeze=True, status=True,
        angle=120, angle0=0):
    """Read XML files with coupling in displacement basis from QE (nosym)."""

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

                    k1 = int(round(np.dot(k, a1) * nk)) % nk
                    k2 = int(round(np.dot(k, a2) * nk)) % nk

                    goto("<PARTIAL_ELPH ")

                    for n in band_select:
                        for m in band_select:
                            if n < 0 or m < 0:
                                next(data)
                            else:
                                my_elph[my_iq, my_irep, k1, k2, n, m] = complex(
                                    *list(map(float, next(data).split(","))))

    comm.Allgatherv(my_elph, (elph,
        sizes * len(rep) * nk * nk * len(bands) * len(bands)))

    return elph[..., 0, 0] if len(bands) == 1 and squeeze else elph

def write_xml_files(filename, data, angle=120, angle0=0):
    """Write XML files with coupling in displacement basis."""

    if comm.rank != 0:
        return

    if data.ndim == 4:
        data = data[:, :, np.newaxis, np.newaxis, :, :]

    nQ, nb, nk, nk, nbands, nbands = data.shape

    a1, a2 = bravais.translations(angle, angle0)
    b1, b2 = bravais.reciprocals(a1, a2)

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

                        k = (k1 * b1 + k2 * b2) / nk

                        xml.write("""
    <K_POINT.%d>
      <COORDINATES_XK type="real" size="3" columns="3">
%23.15E %23.15E %23.15E
      </COORDINATES_XK>
      <PARTIAL_ELPH type="complex" size="%d">"""
                            % (ik, k[0], k[1], 0.0, nbands * nbands))

                        for i in range(nbands):
                            for j in range(nbands):
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
        complex_data = { 'R': False, 'C': True }[columns[-1]]

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
