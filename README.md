# elphmod

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

    python3 -m venv elphmod.venv
    source elphmod.venv/bin/activate
    python3 -m pip install --upgrade pip setuptools wheel
    python3 -m pip install git+https://github.com/janberges/elphmod

elphmod can optionally be run in parallel via MPI (with shared-memory support).
Using APT and pip, you can install the corresponding dependencies as follows:

    sudo apt install libopenmpi-dev
    python3 -m pip install mpi4py --no-binary=mpi4py

If you plan to work on elphmod itself, we recommend to download the complete
repository and install all requirements (including those of documentation and
examples) and a link to the repository in your home directory:

    git clone https://github.com/janberges/elphmod
    python3 -m pip install --user -r elphmod/requirements.txt
    python3 -m pip install --user -e elphmod

Please note that scripts are still copied rather than linked. To circumvent
this, you can alternatively install elphmod by prepending the absolute paths to
`elphmod/elphmod` and `elphmod/bin` to the environmental variables `PYTHONPATH`
and `PATH`, respectively.

## Documentation

All functions are documented directly in the source files using NumPy-style
docstrings. You can generate an automatic documentation in HTML format using
Sphinx:

    cd doc
    make html
    firefox html/index.html

Please also have a look at the examples:

    cd examples
    mpirun python3 electrons.py
    mpirun python3 phonons.py

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

Copyright (C) 2021 elphmod Developers
