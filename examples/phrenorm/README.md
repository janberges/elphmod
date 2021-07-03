# Phonon renormalization in 1H-TaS2

This example shows how to

* perform a cDFPT calculation [Nomura & Arita, PRB 92, 245108 (2015)],
* work with electrons, phonons, and their coupling in localized bases,
* consistently renormalize phonon dispersions (go from cDFPT to DFPT).

For the cDFPT part, you need a modified version of Quantum ESPRESSO. You can
use the provided [patches](../../patches) to apply the required changes.

The results obtained in this example are very far from converged! In return,
the ab initio part takes less than a minute on a laptop with only two cores.

After the main calculation, three further scripts show how to

* extract the deformation potential from the output of the EPW code,
* map the electron-phonon coupling from the unit cell to a supercell,
* check if the hopping parameters from Wannier90 decay appropriately.
