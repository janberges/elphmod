# Patches for Quantum ESPRESSO

Some parts of elphmod (including examples) process the output of a modified
version of Quantum ESPRESSO (QE). This directory contains the corresponding
patches for different versions of QE. To apply them, go to the QE repository
and run, e.g., the following commands:

    git checkout qe-6.7MaX-Release
    git apply /path/to/elphmod/patches/qe-6.7MaX-Release.patch
    ./configure
    make pw pp ph epw

The patch files have been created using `git diff -U1`.
