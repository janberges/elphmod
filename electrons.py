#/usr/bin/env python

import numpy as np
import numpy.linalg

def hamiltonian(hr):
    """Read '_hr.dat' file from Wannier 90 and set up Hamilton operator."""

    with open(hr) as data:
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

        # read lattice vectors and hopping constants:

        cells = np.empty((parameters, 3), dtype=int)
        const = np.empty((parameters, 3, 3), dtype=complex)

        for n in range(parameters):
            tmp = cols()

            cells[n] = list(map(int, tmp[:3]))
            const[n, int(tmp[3]) - 1, int(tmp[4]) - 1] = (
                float(tmp[5]) + 1j * float(tmp[6])) / degeneracy[n // size]

    # return function to calculate hamiltonian for arbitrary k points:

    def calculate_hamiltonian(k1=0, k2=0, k3=0):
        k = np.array([k1, k2, k3])
        H = np.zeros((num_wann, num_wann), dtype=complex)

        for R, C in zip(cells, const):
            H += C * np.exp(1j * np.dot(R, k))

        return H

    return calculate_hamiltonian

def energies(hamiltonian):
    """Calculate electronic energies."""

    return numpy.linalg.eigvalsh(hamiltonian)

def energies_and_states(hamiltonian):
    """Calculate electronic energies and states."""

    return numpy.linalg.eigh(hamiltonian)

def band_order(eps, psi):
    """Sort bands by similarity of eigenstates at neighboring k points."""

    N, bands = eps.shape

    order = np.empty((N, bands), dtype=int)

    n0 = 0
    order[n0] = range(bands)

    for n in range(1, N):
        for i in range(bands):
            order[n, i] = max(range(bands), key=lambda j: np.absolute(
                np.dot(psi[n0, :, order[n0, i]], psi[n, :, j].conj())))

        if np.all(np.absolute(np.diff(eps[n])) > 1e-10): # no degeneracy?
            n0 = n

    return order

def read_bands(filband):
    """Read bands from 'filband' just like Quantum ESRESSO's 'plotband.x'."""

    with open(filband) as data:
        header = next(data)

        # &plot nbnd=  13, nks=  1296 /
        nbnd = int(header[12:16])
        nks  = int(header[22:28])

        bands = np.empty((nbnd, nks))

        for ik in range(nks):
            next(data)

            for lower in range(0, nbnd, 10):
                bands[lower:lower + 10, ik] \
                    = list(map(float, next(data).split()))

    return bands

def read_Fermi_level(pw_scf_out):
    """Read Fermi level from output of self-consistent PW run."""

    with open(pw_scf_out) as data:
        for line in data:
            if 'Fermi energy' in line:
                eF = float(line.split()[-2])
    return eF
