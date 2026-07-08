#!/usr/bin/env python3

# Copyright (C) 2017-2026 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod.models.tas2
import matplotlib.pyplot as plt
import numpy as np
import phonopy
import scipy

el, ph, elph = elphmod.models.tas2.create(rydberg=True, divide_mass=False)

elph = elph.supercell(3, 3, 1)

driver = elphmod.md.Driver(elph, kT=0.001, f='fd', kT0=0.02, f0='mv', nk=(5, 5),
    n=elph.el.size)

driver.random_displacements(amplitude=0.05, reproducible=True)

driver.plot(interactive=True)

scipy.optimize.minimize(driver.free_energy, driver.u, jac=driver.jacobian,
    method='BFGS', options=dict(gtol=1e-8, norm=np.inf))

driver.plot(interactive=False, filename='phonopy_phonons_1.png')

path = 'GMKG'
q, x, corners = elphmod.bravais.path(path, ibrav=4, N=300,
    celldm=np.linalg.norm(elph.ph.a, axis=1))

w1 = elphmod.ph.sgnsqrt(elphmod.dispersion.dispersion(driver.phonons().D, q))

atoms_phonopy = phonopy.structure.atoms.PhonopyAtoms(
    symbols=elph.ph.atom_order,
    positions=driver.r * elphmod.misc.a0,
    cell=elph.ph.a * elphmod.misc.a0,
    masses=elph.ph.M / elphmod.misc.uRy
)

ph_phonopy = phonopy.Phonopy(
    unitcell=atoms_phonopy,
    primitive_matrix=np.eye(3),
    supercell_matrix=np.eye(3),
)

ph_phonopy.generate_displacements(distance=0.01)

forces = []

for supercell in ph_phonopy.supercells_with_displacements:
    driver.r = supercell.positions / elphmod.misc.a0
    forces.append(-driver.jacobian().reshape((-1, 3))
        * elphmod.misc.Ry / elphmod.misc.a0)

ph_phonopy.forces = forces
ph_phonopy.produce_force_constants(calculate_full_force_constants=False)

ph_phonopy.run_band_structure(*phonopy.phonon.band_structure
    .get_band_qpoints_and_path_connections([q[corners] / (2 * np.pi)],
        npoints=10))

w2 = elphmod.ph.sgnsqrt(elphmod.dispersion.dispersion(
    elphmod.ph.Model(ph_phonopy=ph_phonopy).D, q))

if elphmod.MPI.comm.rank == 0:
    for nu in range(elph.ph.size):
        plt.plot(x, w1[:, nu] * elphmod.misc.Ry * 1e3, 'k',
            label=None if nu else 'elphmod')

    for nu in range(elph.ph.size):
        plt.plot(x, w2[:, nu] * elphmod.misc.Ry * 1e3, 'y:',
            label=None if nu else 'phonopy+elphmod')

    for i in range(len(ph_phonopy.band_structure.distances)):
        for nu in range(elph.ph.size):
            plt.plot(
                ph_phonopy.band_structure.distances[i],
                ph_phonopy.band_structure.frequencies[i][:, nu]
                    * 1e3 * elphmod.misc.THz, 'kx',
                label=None if i or nu else 'phonopy')

    plt.ylabel('Phonon energy (meV)')
    plt.xlabel('Wave vector')
    plt.xticks(x[corners], path)
    plt.legend(ncols=3)
    plt.savefig('phonopy_phonons_2.png')
    plt.show()
