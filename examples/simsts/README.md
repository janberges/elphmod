# Simulated STS spectrum of graphene

This example shows how to

* work with Wannier functions in the position representation from Wannier90,
* simulate scanning-tunnelling microscopy (STM) and spectroscopy (STS) data.

The Wannier functions used to represent the Hamiltonian (*seedname_hr.dat*) and
those written to file (*seedname_0000X.xsf*) must be identical (including sign).
Hence, the following line in *plot.F90* of Wannier90 must be commented out:

    wann_func(:, :, :, loop_w) = wann_func(:, :, :, loop_w)/wmod

The scanning-tunnelling spectrum is calculated just like the density of states,
except that each state is weighted with the square modulus of its overlap with
an s orbital located at a chosen position above the sample.
