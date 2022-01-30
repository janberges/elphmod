[![DOI](https://zenodo.org/badge/330427606.svg)](https://zenodo.org/badge/latestdoi/330427606)

![elphmod logo](https://raw.githubusercontent.com/janberges/elphmod/master/logo/logo.svg)

This is a collection of Python modules to handle electron-phonon models:

* `el` - tight-binding models from Wannier90
* `ph` - mass-spring models from Quantum ESPRESSO
* `elph` - electron-phonon coupling from EPW
* `elel` - Coulomb interaction from VASP
* `MPI` - work distribution and shared memory
* `bravais` - lattices, symmetries, and interpolation
* `dispersion` - diagonalization on paths and meshes
* `dos` - 2D tetrahedron methods
* `diagrams` - susceptibilities, self-energies, etc.
* `occupations` - step and delta smearing functions
* `eliashberg` - parameters for McMillan's formula
* `plot` - BZ plots, color models, and fatbands
* `misc` - status bars etc.

## Installation

To install the latest version of elphmod in a virtual environment:

    python3 -m venv venv
    source venv/bin/activate
    python3 -m pip install elphmod

Alternatively, to install the latest development version:

    python3 -m pip install --upgrade pip setuptools wheel
    python3 -m pip install git+https://github.com/janberges/elphmod

elphmod can optionally be run in parallel via MPI (with shared-memory support).
Using APT and pip, you can install the corresponding dependencies as follows:

    sudo apt install libopenmpi-dev
    python3 -m pip install mpi4py --no-binary=mpi4py

If you plan to work on elphmod itself, we recommend to download the complete
repository and also install the requirements of examples and documentation:

    git clone https://github.com/janberges/elphmod
    python3 -m pip install -r elphmod/requirements.txt

Also consider adding the absolute paths to `elphmod/elphmod` and `elphmod/bin`
to the environment variables `PYTHONPATH` and `PATH`, respectively.

## Documentation

The documentation can be found at <https://janberges.github.io/elphmod>. It has
been automatically generated from the NumPy-style docstrings in the source code
using Sphinx:

    cd doc
    make html
    firefox html/index.html

Please also have a look at the examples:

    cd examples
    mpirun python3 electrons.py

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

Copyright (C) 2017-2022 elphmod Developers
