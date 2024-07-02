# elphmod examples

This directory contains examples that demonstrate the different functionalities
of elphmod. To run them, the scripts `run.sh` can be used.

The versions of the installed elphmod and of the examples must match. To ensure
this, go to the appropriate revision of this repository:

    git checkout v`python3 -c 'import elphmod; print(elphmod.__version__)'`

Also make sure to install the Python requirements before:

    python3 -m pip install -r requirements.txt

(Installing PyQt5 is only necessary if the Tk interface is not available.)

* `a2f.py` - Eliashberg spectral function of monolayer TaS2
* `coulomb.py` - Coulomb interaction from VASP
* `dos.py` - density of states via smearing and tetrahedron methods
* `double_delta.py` - double-delta integral via smearing and tetrahedron methods
* `electrons.py` - electronic band structure and density of states
* `fan_migdal.py` - electron renormalization from Fan-Migdal self-energy
* `fermi_surface.py` - Fermi surface via smearing method
* `fermi_velocity.py` - Fermi velocity on whole Brillouin zone
* `intersections.py` - Fermi-surface intersections via tetrahedron method
* `irreducibles.py` - inequivalent nth neighbors of triangular lattice
* `isoline.py` - Fermi surface via tetrahedron method
* `kekule.py` - relaxation of Kekulé distortion in strongly coupled graphene
* `occupations.py` - occupation function, delta function, generalized entropy
* `peierls.py` - band unfolding for the example of Peierls trimerization
* `phonons.py` - fatband plot of phonon dispersion compared to QE reference
* `skiing.py` - atomic displacements along combination of unstable phonon modes
* `smear_tetra.py` - quantitative comparison of smearing and tetrahedron methods
* `specfun.py` - phonon spectral function via phonon self-energy correction
* `ssh.py` - relaxation of Peierls distortion in SSH model
* `supercell_el.py` - supercell mapping and unfolding for electrons
* `supercell_elel.py` - supercell mapping and unfolding for el.-el. interaction
* `supercell_elph.py` - supercell mapping and unfolding for el.-ph. interaction
* `supercell_ph.py` - supercell mapping and unfolding for phonons
* `susceptibility.py` - bare electronic susceptibility with hybridization
* `tetrahedron.py` - some computer algebra behind tetrahedron method

More sophisticated examples have their own subdirectory. Most of them require a
[patched](../patches) version of Quantum ESPRESSO.

* `bare` - bare phonon dispersion
* `cdw_1d` - downfolding and CDW relaxation for carbon chain
* `fluctuations` - fluctuation diagnostics of phonon anomalies in 1H-TaS2
* `goldstone` - Goldstone modes of nitrogen molecule
* `lambda` - electron-phonon coupling for superconductivity
* `md` - interface with i-PI code for molecular-dynamics simulations
* `modes` - decomposition of CDW modes into harmonic eigenmodes from DFPT
* `phrenorm` - phonon renormalization in 1H-TaS2
* `phrenorm_3d` - phonon renormalization in polonium
* `phrenorm_graphene` - phonon renormalization in graphene
* `ph_vs_epw` - comparison of couplings from PHonon and EPW
* `projwfc_1d` - projected band structure of carbon chain
* `projwfc` - projected band structure of graphene
* `projwfc_3d` - projected band structure of polonium
* `quadrupole` - optimization of quadrupole tensor for bare phonons of 1H-TaS2
* `simstm` - simulated STM image of 1H-TaS2
* `simsts` - simulated STS spectrum of graphene
* `wannier` - Wannier interpolation in 1H-TaS2
