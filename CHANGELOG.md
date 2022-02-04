# Changes

* Fixed sign in Marzari-Vanderbilt-De Vita-Payne smearing
* Interatomic distances are stored together with force constants

## v0.10

* More forgiving namelist input
* QE-7.0 support (cDFPT patch)

## v0.9

* New electron-electron class (incl. supercell mapping)
* Shared-memory array as subclass of NumPy array
* Improved handling of XCrySDen and Gaussian cube files

## v0.8

* QE patches included in source distribution
* Flexible basis/motif of tight-binding and mass-spring models
* Improved selection of cDFPT target subspaces

## v0.7

* Faster Fourier transforms (Hamiltonian, dynamical matrix, coupling)
* QE-6.8 support (cDFPT patch, optimal Wigner-Seitz weights in EPW)
* Wannier functions in position representation as part of TB model

## v0.6

* Mapping of all models to arbitrary commensurate supercells
* New functions to read and write input of QE phonon codes
* New Bash script to prevent NumPy's auto-parallelization
* Fixed ordering of eigenvectors calculated on 2D mesh

## v0.5

* Examples run successfully without `mpi4py`

## v0.4

* New QE input parameters `cdfpt_bnd` and `cdfpt_orb`
* QE modifications provided as Git patches instead of scripts
* Mapping of tight-binding and mass-spring models onto supercell

## v0.3

* Support for `_wsvec.dat` files
* Several functions generalized to 3D
* Improved handling of Bravais lattices

## v0.2

* `pip install elphmod` installs helper scripts

## v0.1

* Initial pre-release
