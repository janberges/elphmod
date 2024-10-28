# Decomposition of charge-density-wave (CDW) modes into harmonic eigenmodes from DFPT

This example is still work in progress,
but at the moment we can insert
- the CDW supercell dimensions
- a CDW structure as a Quantum ESPRESSO input file,
- the associated harmonic interatomic force constants from DFPT as an .ifc file

and the code aligns the CDW and the symmetric structure.

Returns the file: info.dat

Open issues:
- it does not work for rotated structures
