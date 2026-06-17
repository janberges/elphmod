# Contributing

Patches, merge request, issue reports, and feature requests are most welcome,
either via email or through any of the web platforms where elphmod is hosted.

## Code checks

The following tasks should be performed regularly and especially before
publishing a new version.

* Reformat source code with ``frettipy`` and manually wrap overlong lines.
* Validate source code using ``flake8`` and fix all errors and warnings.
* Run tests both serially and in parallel for different Python versions.
* Check if all examples work for the latest QE patch and i-PI installed.
* Update documentation (which should also reveal changes in the examples).

## To do

The following things should/could be done sooner or later. Items with question
mark are not necessarily a good idea.

* Remember order of atomic species in header of force-constant files.
* Use generic (argument agnostic) function to write all QE input files.
* Disentangle two-dimensional bands with spin degree of freedom.
* Check if cDFPT examples work for for `nspin = 2` or `noncolin = .true.`
* Shift Wannier functions more accurately in STM and STS simulations.
* Move I/O functions and numerical constants to separate modules.
* Free local memory after data has been gathered in global memory.
* Replace status arguments etc. by global `elphmod(.misc).verbosity`.
