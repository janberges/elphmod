# Python modules for electron-phonon models

elphmod is a collection of Python modules to handle coupled tight-binding and
mass-spring models derived from first principles. It provides interfaces with
popular simulation software such as Quantum ESPRESSO, Wannier90, EPW, RESPACK,
and i-PI. It helps calculate dispersions, spectra, and response functions and
can be used to build and study distorted structures on supercells.

* `el` - tight-binding models from Wannier90
* `ph` - mass-spring models from Quantum ESPRESSO
* `elph` - electron-phonon coupling from EPW
* `elel` - Coulomb interaction from RESPACK
* `MPI` - work distribution and shared memory
* `bravais` - lattices, symmetries, and interpolation
* `dispersion` - diagonalization on paths and meshes
* `dos` - 2D tetrahedron methods
* `diagrams` - susceptibilities, self-energies, etc.
* `occupations` - step and delta smearing functions
* `md` - charge-density-wave dynamics using i-PI
* `eliashberg` - parameters for McMillan's formula
* `plot` - BZ plots, fatbands, etc.
* `misc` - constants, status bars, parsing, etc.
* `models` - nearest-neighbor models for testing

## Installation

You can install the latest version of elphmod from PyPI:

    python3 -m pip install elphmod

Or from the conda-forge channel on Anaconda Cloud:

    conda install conda-forge::elphmod

elphmod can optionally be run in parallel via MPI (with shared-memory support).
Using APT and pip, you can install the corresponding dependencies as follows:

    sudo apt install libopenmpi-dev
    python3 -m pip install mpi4py --no-binary=mpi4py

You can also download the complete repository, perform an editable installation,
and install the requirements of examples and documentation:

    git clone https://github.com/janberges/elphmod
    python3 -m pip install -e elphmod
    python3 -m pip install -r elphmod/examples/requirements.txt
    python3 -m pip install -r elphmod/doc/requirements.txt

## Documentation

The documentation can be found at <https://io.janberges.de/elphmod>.

Please also have a look at the examples directory.

## Reference

elphmod is stored on Zenodo: <https://doi.org/10.5281/zenodo.5919991>.

## Licence

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.

Copyright (C) 2017-2025 elphmod Developers
