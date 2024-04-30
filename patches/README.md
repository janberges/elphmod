# Patches for Quantum ESPRESSO

Some parts of elphmod (including examples) process the output of a modified
version of Quantum ESPRESSO (QE). This directory contains the corresponding
patches for different versions of QE. To apply them, go to the QE repository
and run, e.g., the following commands:

    git checkout qe-7.3
    git apply /path/to/elphmod/patches/qe-7.3.patch
    ./configure
    make pw pp ph epw

The patch files have been created using `git diff -U1`.

The following description applies to `qe-6.7MaX-Release` and higher versions.
The patch for `qe-6.3-backports` uses different formats for input and output,
which requires using `elphmod.elph.Model(old_ws=True)`.

## Changes in PHonon

The following inputs are used for constrained density-functional perturbation
theory (cDFPT) [Nomura and Arita, Phys. Rev. B **92**, 245108 (2015)]. If not
set, a standard DFPT calculation is done.

- `cdfpt_bnd`: Definition of cDFPT active subspace via list of band indices.
  The band indices are defined by QE, where bands are sorted by energy.
- `cdfpt_orb`: Definition of cDFPT active subspace via list of orbital indices.
  This requires that a `projwfc.x` calculation for the complete uniform _k_ mesh
  used in the phonon calculation is done before. The orbital indices are defined
  in the output file of `projwfc.x`. If _n_ orbitals are specified, for each _k_
  point, the _n_ states with the largest projection onto the subspace spanned by
  these _n_ orbitals are selected. If `cdfpt_bnd` is also specified, up to _n_
  states among these bands are chosen.
- `bare`: Suppress electronic response to atomic displacements? This corresponds
  to a cDFPT active subspace containing _all_ states.

Additionally, the option `electron_phonon = 'defpot'` prints the electron-phonon
matrix elements in Ry/bohr in the Cartesian basis to standard output.

## Changes in EPW

If `epwwrite` is enabled, the file `wigner.fmt` is written. It contains the
Wigner-Seitz vectors and degeneracies corresponding to the electron-phonon
matrix elements in the localized representation in `outfir/prefix.epmatwp`.
