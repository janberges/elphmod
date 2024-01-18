# To do

The following things should/could be done sooner or later. Items with question
mark are not necessarily a good idea.

* Unscreen phonons of MD driver on supercell (using sparse matrices).
* Set `ph.D0` and `elph.g0` when generating models with `q2r`.
* Read Wannier-function centres from `_centres.xyz`.
* Add long-range argument `lr` to electron-phonon class `elph`?
* Save `_wsvec.dat` together with `_hr.dat` file (important for small meshes).
* Remember order of atomic species in header of force-constant files.
* Use generic (argument agnostic) function to write all QE input files.
* Use `assume_isolated = '2D'` in all examples with 2D systems.
* Consider that minima and maxima of bitmaps fall on pixel edges, not centers.
* Use info function from MPI module to raise errors and warnings.
* Disentangle two-dimensional bands with spin degree of freedom.
* Check if cDFPT examples work for for `nspin = 2` or `noncolin = .true.`
* Shift Wannier functions more accurately in STM and STS simulations.
* Save `el.Model`, `ph.Model`, and `elph.Model` to files (`_hr.dat`, `flfrc`).
* Move I/O functions and numerical constants to separate modules.
* Free local memory after data has been gathered in global memory.
* Drop Python-2 support in favor of things like `(nq, nph, *nk, nel, nel)`?
* Replace status arguments etc. by global `elphmod(.misc).verbosity`.
