#/usr/bin/env python

import numpy as np

from . import MPI
comm = MPI.comm

class Model(object):
    """Tight-binding model for the electrons."""

    def H(self, k1=0, k2=0, k3=0):
        """Set up Hamilton operator for arbitrary k point."""

        k = np.array([k1, k2, k3])
        H = np.empty(self.data.shape, dtype=complex)

        for n in range(H.shape[0]):
            H[n] = self.data[n] * np.exp(1j * np.dot(self.R[n], k))

        return H.sum(axis=0)

    def __init__(self, hrdat):
        """Prepare hopping parameters."""

        self.R, self.data = read_hrdat(hrdat)
        self.size = self.data.shape[1]

def read_hrdat(hrdat):
    """Read '_hr.dat' file from Wannier90."""

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

        degeneracy = []

        while len(degeneracy) < nrpts:
            degeneracy.extend(map(float, cols()))
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
                    float(tmp[5]) + 1j * float(tmp[6])) / degeneracy[n]

            cells[n] = list(map(int, tmp[:3]))

        data.close()

    comm.Bcast(cells)
    comm.Bcast(const)

    return cells, const

def read_bands(filband):
    """Read bands from 'filband' just like Quantum ESRESSO's 'plotband.x'."""

    if comm.rank == 0:
        data = open(filband)

        header = next(data)

        # &plot nbnd=  13, nks=  1296 /
        _, nbnd, nks = header.split('=')
        nbnd = int(nbnd[:nbnd.index(',')])
        nks  = int( nks[: nks.index('/')])
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
    """Read bands from 'filband.gnu' produced by Quantum ESPRESSO's 'bands.x'.

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
        File with standard output from Quantum ESPRESSO's 'bands.x'.

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

def read_atomic_projections(atomic_proj_xml):
    """Read projected bands from 'outdir/prefix.save/atomic_proj.xml'."""

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
    proj = np.empty((nk, bands, no))

    if comm.rank == 0:
        goto('<K-POINTS ')
        for i in range(nk):
            k[i] = list(map(float, next(data).split()))

        for i in range(nk):
            goto('<EIG ')
            for n in range(bands):
                eps[i, n] = float(next(data))

        for ik in range(nk):
            for a in range(no):
                goto('<ATMWFC.')
                for n in range(bands):
                    Re, Im = list(map(float, next(data).split(',')))
                    proj[ik, n, a] = Re * Re + Im * Im

        data.close()

        x[0] = 0

        for i in range(1, nk):
            dk = k[i] - k[i - 1]
            x[i] = x[i - 1] + np.sqrt(np.dot(dk, dk))

        eps -= mu

    comm.Bcast(x)
    comm.Bcast(k)
    comm.Bcast(eps)
    comm.Bcast(proj)

    return x, k, eps, proj

def read_Fermi_level(pw_scf_out):
    """Read Fermi level from output of self-consistent PW run."""

    if comm.rank == 0:
        with open(pw_scf_out) as data:
            for line in data:
                if 'Fermi energy' in line:
                    eF = float(line.split()[-2])
    else:
        eF = None

    eF = comm.bcast(eF)

    return eF
