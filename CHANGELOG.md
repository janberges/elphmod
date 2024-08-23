# Changelog

See [Git commits](https://github.com/janberges/elphmod/commits) for more
detailed list of changes.

## v0.28/2024-08-23

* Support for `mpi4py` version 4.0.0
* Loading of *.xyz* files (needed for i-PI example) was broken in version 0.27

## v0.27/2024-07-23

* Faster calculation of long-range terms
* Syntax upgrade (absolute imports, Python 3)
* New function to define uniform meshes
* Fixed methods to shift orbitals and atoms
* Various new unit tests

## v0.26/2024-06-17

* Small but relevant bugfix in new EPW interface

## v0.25/2024-06-14

* Superconductivity calculations on large supercells
* Improved documentation (patches, conventions, source-code links)
* New format for Wigner-Seitz data (*wigner.fmt*)
* Fan-Migdal self-energy and Green-Kubo conductivity
* Size-consistent Ewald parameter (as in development version of QE)
* Allen-Dynes formula with strong-coupling and shape corrections
* Improved handling of atomic projections (phase, spin-orbit coupling)

## v0.24/2024-04-03

* Improved MD driver and i-PI example
* Calculation of band velocity without finite differences
* Documentation of indices of matrix elements
* Export of coupling in EPW format (*.epmatwp*)
* Transformation of coupling to arbitrary real-space meshes
* Cleanup of routines related to Wigner-Seitz cell
* Subpackage `elphmod.models` for nearest-neighbor model input data

## v0.23/2024-03-07

* QE-7.3 support (cDFPT patch, no longer included in source distribution)
* pyproject.toml-based build with reviewed requirements
* Scripts replaced by entry points (for editable installs and portability)
* Phonons and faster forces from MD driver using sparse coupling matrices
* Official PNG version of logo
* Read the Docs theme
* First unit test
* Various bugfixes and optimizations

## v0.22/2023-10-14

* Import of Wannier Hamiltonian from RESPACK
* Export of dynamical matrices from MD driver

## v0.21/2023-09-20

* Support for more QE Bravais lattices
* Support for electron-electron interaction beyond 2D systems
* Consistent mapping of electron-electron interaction to supercells
* Export of electron-electron interaction in RESPACK format (*dat.Wmat*)
* Option to suppress status bars

## v0.20/2023-06-22

* Bugfix and example related to Eliashberg theory
* Model input data (electrons, phonons, and coupling) for TaS₂
* Faster mapping to supercells
* Tight-binding models from perturbed states on supercells
* Export of tight-binding model in Wannier90 format (*_hr.dat*)
* QE-7.2 support (cDFPT patch)
* Dynamical matrices and force constants from XML files

## v0.19/2023-03-08

* Several bugfixes related to supercells and long-range components
* Experimental long-range part of sparse dynamical matrix and coupling

## v0.18/2023-02-12

* Electron-phonon matrix elements directly from PHonon code (QE patch)
* Consistent matrix-elements and Fourier-transform convention for phonons
* Export of mass-spring models in QE force-constants format (`flfrc`)
* Export of band structures in QE bands format (`filband`)
* Some bugfixes and optimizations

## v0.17/2023-01-16

* Charge-density-wave molecular dynamics using i-PI
* Animated relaxation of 1D Peierls and 2D Kekulé dimerization
* Experimental acoustic-sum-rule correction for coupling
* Parameters of McMillan's formula from smearing method
* Determination of Fermi level via fixed-point iteration
* Mapping of Born effective charges etc. to supercells
* Decay of electron-phonon matrix elements

## v0.16/2022-09-28

* No MPI overhead for serial runs
* More accurate symmetrization methods
* More efficient setup of large supercells
* Hamiltonian, dynamical matrix, and coupling as sparse matrices
* Jmol's atom "CPK" colors [https://jmol.sourceforge.net/jscolors]
* RPA and cRPA Coulomb interaction from RESPACK
* Example on electron-phonon fluctuation diagnostics
* 2D long-range electrostatics by Poncé et al. [PRB 107, 155424 (2023)]
* Various bugfixes and optimizations

## v0.15/2022-07-29

* Handling of wave functions and charge density from QE
* 1D linear interpolation
* Phonon models from dynamical-matrix files (skip `q2r.x`)
* Long-range terms of dynamical matrices and coupling from cDFPT
* Faster band unfolding
* Calculation of band and entropy contributions from PWscf output
* Generalized entropy for all smearing functions

## v0.14/2022-06-22

* QE-7.1 support (cDFPT patch)
* Module docstrings
* More efficient smearing functions
* Fluctuation diagnostics of triangle diagram
* No nested progress bars
* Changeable long- and corresponding short-range phonons and coupling
* Automatic detection of two-dimensional systems (`lr2d`)

## v0.13/2022-05-23

* Phonon self-energy as a function of *separate* vertices
* Decluttered plot module (some functions now in StoryLines package)
* Support for `_tb.dat` files (tight-binding data from Wannier90)

## v0.12/2022-05-06

* Model input data (electrons, phonons, and coupling) for graphene
* New function to write interatomic force constants in QE format
* Calculation of phonon spectral function
* Calculation of phonon self-energy arising from Fermi-level shift
* Further functions generalized to 3D

## v0.11/2022-03-16

* Symmetrization of Hamiltonian, dynamical matrix, and coupling
* Long-range dipole and quadrupole terms of dynamical matrix and coupling
* 1st and 3rd order of diagrammatic expansion of grand potential
* Derivatives of delta functions
* Mapping to supercells with "negative volume"
* Fixed sign in Marzari-Vanderbilt-De Vita-Payne smearing
* Interatomic distances are stored together with force constants

## v0.10/2022-01-30

* More forgiving namelist input
* QE-7.0 support (cDFPT patch)

## v0.9/2022-01-06

* New electron-electron class (incl. supercell mapping)
* Shared-memory array as subclass of NumPy array
* Improved handling of XCrySDen and Gaussian cube files

## v0.8/2021-11-02

* QE patches included in source distribution
* Flexible basis/motif of tight-binding and mass-spring models
* Improved selection of cDFPT target subspaces

## v0.7/2021-10-13

* Faster Fourier transforms (Hamiltonian, dynamical matrix, coupling)
* QE-6.8 support (cDFPT patch, optimal Wigner-Seitz weights in EPW)
* Wannier functions in position representation as part of TB model

## v0.6/2021-09-21

* Mapping of all models to arbitrary commensurate supercells
* New functions to read and write input of QE phonon codes
* New Bash script to prevent NumPy's auto-parallelization
* Fixed ordering of eigenvectors calculated on 2D mesh

## v0.5/2021-07-20

* Examples run successfully without `mpi4py`

## v0.4/2021-07-09

* New QE input parameters `cdfpt_bnd` and `cdfpt_orb`
* QE modifications provided as Git patches instead of scripts
* Mapping of tight-binding and mass-spring models onto supercell

## v0.3/2021-06-11

* Support for `_wsvec.dat` files
* Several functions generalized to 3D
* Improved handling of Bravais lattices

## v0.2/2021-05-10

* `pip install elphmod` installs helper scripts

## v0.1/2021-05-07

* Initial pre-release
