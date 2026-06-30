#!/usr/bin/env python3

# Copyright (C) 2017-2026 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.
# The FCC-Al example stems from https://docs.ase-lib.org/ase/phonons.html.

import ase.build
import ase.calculators.emt
import ase.phonons
import elphmod
import matplotlib.pyplot as plt
import numpy as np

a = 4.05
atoms = ase.build.bulk('Al', 'fcc', a=a)

ph_ase = ase.phonons.Phonons(atoms, ase.calculators.emt.EMT(),
    supercell=(5,) * 3, delta=0.05, name='ase_phonons')

ph_ase.run()
ph_ase.read(acoustic=True)
ph_ase.clean()

path = 'GXULGK'
path_ase = atoms.cell.bandpath(path, npoints=100)
band_ase = ph_ase.get_band_structure(path_ase, verbose=False)

q, x, corners = elphmod.bravais.path(path, ibrav=2, a=a, N=300)
x *= 2 * np.pi

ph = elphmod.ph.Model(ph_ase=ph_ase)
w2 = elphmod.dispersion.dispersion(ph.D, q)
w = elphmod.ph.sgnsqrt(w2) * elphmod.misc.Ry

if elphmod.MPI.comm.rank == 0:
    band_ase.plot(emin=-0.01, emax=0.04, color='red', label='ASE')
    plt.plot(x, w, 'k--')

    plt.ylabel('Phonon energy (eV)')
    plt.savefig('ase_phonons.png')
    plt.legend()
    plt.show()
