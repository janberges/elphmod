# Goldstone modes

The movement of the atoms of a diatomic molecule can be classified into the
following six eigenmodes:

* one bond-stretching mode,
* three translational modes,
* two rotational modes.

Only the first mode has a nonzero vibration frequency. Since the presence of
the molecule breaks the homogeneity and isotropy (except for a rotation about
the bond axis) of the space (or: since the total energy must be invariant with
respect to a rigid translation or rotation of the molecule), the remaining five
modes are Goldstone bosons with zero energy (or: without any restoring force).

Using the example of a nitrogen molecule, this example shows that

* cDFPT phonons do not always satisfy the Goldstone theorem [van Loon et al.,
  Phys. Rev. B 103, 205103 (2021)],
* the acoustic sum rule correction restores the translational Goldstone modes,
* the Born-Huang sum rule correction restores the rotational Goldstone modes.

The example also shows how to calculate exact DFPT phonon self-energies for
gapped systems (i.e., without any electronic occupation smearing).

For the (c)DFPT part, you need a modified version of Quantum ESPRESSO. You
can use the provided patches to apply the required changes.

The results obtained in this example are far from converged!
