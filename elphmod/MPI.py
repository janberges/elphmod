#/usr/bin/env python

import numpy as np
from mpi4py import MPI

def shared_array(comm, shape, dtype):
    "Create array whose memory is shared among all processes."

    # Shared memory allocation following Lisandro Dalcin on Google Groups:
    # 'Shared memory for data structures and mpi4py.MPI.Win.Allocate_shared'

    size = np.prod(shape)
    dtype = np.dtype(dtype)
    itemsize = dtype.itemsize

    if comm.rank == 0:
        bytes = size * itemsize
    else:
        bytes = 0

    win = MPI.Win.Allocate_shared(bytes, itemsize, comm=comm)
    buf, itemsize = win.Shared_query(0)
    buf = np.array(buf, dtype='B', copy=False)

    return np.ndarray(shape, buffer=buf, dtype=dtype)
