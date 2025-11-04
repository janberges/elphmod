# Contributing

Patches, merge request, issue reports, and feature requests are most welcome,
either via email or through any of the web platforms where elphmod is hosted.

## Code checks

The following checks should be performed regularly and especially before
publishing a new version.

* Check if all examples work for the latest QE patch and i-PI installed.
* Update documentation (which should also reveal changes in the examples).
* Run tests both serially and in parallel for different Python versions.
* Reformat source code with ``frettipy`` and manually wrap overlong lines.

## To do

The following things should/could be done sooner or later. Items with question
mark are not necessarily a good idea.

* Use size-consistent tolerance for electron number when finding Fermi level?
* Set `ph.D0` and `elph.g0` when generating models with `q2r`?
* Read Wannier-function centres from `_centres.xyz`.
* Add long-range argument `lr` to electron-phonon class `elph`?
* Remember order of atomic species in header of force-constant files.
* Use generic (argument agnostic) function to write all QE input files.
* Use `assume_isolated = '2D'` in all examples with 2D systems.
* Consider that minima and maxima of bitmaps fall on pixel edges, not centers.
* Use info function from MPI module to raise errors and warnings.
* Disentangle two-dimensional bands with spin degree of freedom.
* Check if cDFPT examples work for for `nspin = 2` or `noncolin = .true.`
* Shift Wannier functions more accurately in STM and STS simulations.
* Move I/O functions and numerical constants to separate modules.
* Free local memory after data has been gathered in global memory.
* Replace status arguments etc. by global `elphmod(.misc).verbosity`.
