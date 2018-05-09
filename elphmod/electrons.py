#/usr/bin/env python

import numpy as np

def hamiltonian(comm, hr):
    """Read '_hr.dat' file from Wannier 90 and set up Hamilton operator."""

    if comm.rank == 0:
        data = open(hr)

        # read all words of current line:

        def cols():
            return data.readline().split()

        # skip header:

        date = cols()[2]

        # get dimensions:

        num_wann = int(cols()[0])
        nrpts = int(cols()[0])

        size = num_wann ** 2
        parameters = size * nrpts

        # read degeneracies of Wigner-Seitz grid points:

        degeneracy = []

        while len(degeneracy) < nrpts:
            degeneracy.extend(map(float, cols()))
    else:
        num_wann = parameters = None

    num_wann = comm.bcast(num_wann)
    parameters = comm.bcast(parameters)

    cells = np.empty((parameters, 3), dtype=int)
    const = np.empty((parameters, 3, 3), dtype=complex)

    if comm.rank == 0:
        # read lattice vectors and hopping constants:

        for n in range(parameters):
            tmp = cols()

            cells[n] = list(map(int, tmp[:3]))
            const[n, int(tmp[3]) - 1, int(tmp[4]) - 1] = (
                float(tmp[5]) + 1j * float(tmp[6])) / degeneracy[n // size]

        data.close()

    comm.Bcast(cells)
    comm.Bcast(const)

    # return function to calculate hamiltonian for arbitrary k points:

    def calculate_hamiltonian(k1=0, k2=0, k3=0):
        k = np.array([k1, k2, k3])
        H = np.zeros((num_wann, num_wann), dtype=complex)

        for R, C in zip(cells, const):
            H += C * np.exp(1j * np.dot(R, k))

        return H

    calculate_hamiltonian.size = num_wann

    return calculate_hamiltonian

def read_bands(filband):
    """Read bands from 'filband' just like Quantum ESRESSO's 'plotband.x'."""

    with open(filband) as data:
        header = next(data)

        # &plot nbnd=  13, nks=  1296 /
        nbnd = int(header[12:16])
        nks  = int(header[22:28])

        k = np.empty((nks, 3))
        bands = np.empty((nbnd, nks))

        for ik in range(nks):
            k[ik] = list(map(float, next(data).split()))

            for lower in range(0, nbnd, 10):
                bands[lower:lower + 10, ik] \
                    = list(map(float, next(data).split()))

    return k, bands

def read_Fermi_level(pw_scf_out):
    """Read Fermi level from output of self-consistent PW run."""

    with open(pw_scf_out) as data:
        for line in data:
            if 'Fermi energy' in line:
                eF = float(line.split()[-2])
    return eF

def susceptibility(e, T=1.0, eta=1e-3):
    """Calculate real part of static electronic susceptibility

        chi(q) = 2/N sum[k] [f(k+q) - f(k)] / [e(k+q) - e(k) + i eta].

    The resolution in q is limited by the resolution in k."""

    nk, nk = e.shape

    T *= 8.61733e-5 # K to eV

    f = 1 / (np.exp(e / T) + 1)

    e = np.tile(e, (2, 2))
    f = np.tile(f, (2, 2))

    def calculate_susceptibility(q1=0, q2=0):
        q1 = int(round(q1 / (2 * np.pi) * nk)) % nk
        q2 = int(round(q2 / (2 * np.pi) * nk)) % nk

        de = e[q1:q1 + nk, q2:q2 + nk] - e[:nk, :nk]
        df = f[q1:q1 + nk, q2:q2 + nk] - f[:nk, :nk]

        return 2 * np.sum(df * de / (de * de + eta * eta)) / nk ** 2

    return calculate_susceptibility
