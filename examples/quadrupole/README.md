# Optimization of quadrupole tensors

This is a minimal working example of the optimization of quadrupole tensors as
done by J. Berges, N. Girotto, T. Wehling, N. Marzari, and S. Poncé, *Phonon
self-energy corrections: To screen, or not to screen*, [Phys. Rev. X **13**,
041009 (2023)](https://doi.org/10.1103/PhysRevX.13.041009). The original version
can be found [here](https://doi.org/10.24435/materialscloud:he-pv).

More precisely, we first calculate (bare) dynamical matrices and electron-phonon
matrix elements both on a coarse q mesh and for selected q points along a path.
Then we Fourier interpolate the former, minimizing deviations from the latter.
Here, the free parameters are the independent elements of the quadrupole tensors
Q and the range-separation parameter L entering the formulas for the long-range
components that are subtracted and added before and after interpolation. First,
we optimize L for Q = 0, minimizing the short-range part of the force constants.
Second, we optimize Q for constant L. A simultaneous optimization would also be
possible, but it is only important that L is in the correct range.

Note that the results obtained in this example are not converged. The parameters
have been chosen such that the calculations can be done on a personal computer.
