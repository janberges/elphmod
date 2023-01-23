# To do

The following things should/could be done sooner or later. Items with question
mark are not necessarily a good idea.

* Use `assume_isolated = '2D'` in all examples with 2D systems.
* Consider that minima and maxima of bitmaps fall on pixel edges, not centers.
* Use info function from MPI module to raise errors and warnings.
* Disentangle two-dimensional bands with spin degree of freedom.
* Check if cDFPT examples work for for `nspin = 2` or `noncolin = .true.`
* Shift Wannier functions more accurately in STM and STS simulations.
* Save `el.Model`, `ph.Model`, and `elph.Model` to files (`_hr.dat`, `flfrc`).
* Move I/O functions and numerical constants to separate modules.
* Free local memory after data has been gathered in global memory.
* Read (and write?) dynamical matrices and force constants in XML format.
* Drop Python-2 support in favor of things like `(nq, nph, *nk, nel, nel)`?
* Replace status arguments etc. by global `elphmod(.misc).verbosity`.
* Change sign convention in phononic Fourier transform? Many side effects!
