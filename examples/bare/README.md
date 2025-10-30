# Bare phonons

This example will show you how to calculate a bare phonon dispersion, excluding
any response of the electron density to displacements of the (pseudo-)ions. This
is the limit of constrained density-functional perturbation theory (cDFPT) where
all electronic states are in the active subspace. Note the bare Born effective
charges and corresponding long-range terms and the broken acoustic sum rules.

For the bare phonons, you need a modified version of Quantum ESPRESSO. You
can use the provided [patch](../../patches) to apply the required changes.
