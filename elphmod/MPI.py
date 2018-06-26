#/usr/bin/env python

import sys
import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD

def distribute(size, bounds=False):
    """Distribute work among processes."""

    sizes = np.empty(comm.size, dtype=int)

    if comm.rank == 0:
        sizes[:] = size // comm.size
        sizes[:size % comm.size] += 1

    comm.Bcast(sizes)

    if bounds:
        cumsum = np.empty(comm.size + 1, dtype=int)

        if comm.rank == 0:
            cumsum[0] = 0

            for rank in range(comm.size):
                cumsum[rank + 1] = cumsum[rank] + sizes[rank]

        comm.Bcast(cumsum)

        return sizes, cumsum

    return sizes

def shared_array(shape, dtype):
    "Create array whose memory is shared among all processes."

    # Shared memory allocation following Lisandro Dalcin on Google Groups:
    # 'Shared memory for data structures and mpi4py.MPI.Win.Allocate_shared'

    size = np.prod(shape)
    dtype = np.dtype(dtype)
    itemsize = dtype.itemsize

    # From article from Intel Developer Zone
    # 'An Introduction to MPI-3 Shared Memory Programming':

    node = comm.Split_type(MPI.COMM_TYPE_SHARED, key=comm.rank) # same node

    # From Gilles reply to StackOverflow question
    # 'get Nodes with MPI program in C':

    images = comm.Split(node.rank, key=comm.rank) # same node.rank

    if node.rank == 0:
        bytes = size * itemsize
    else:
        bytes = 0

    win = MPI.Win.Allocate_shared(bytes, itemsize, comm=node)
    buf, itemsize = win.Shared_query(0)

    return node, images, np.ndarray(shape, buffer=buf, dtype=dtype)

def info(message, error=False):
    """Print status message from first process."""

    comm.barrier()

    if comm.rank == 0:
        if error:
            sys.stdout.write('Error: ')

        print(message)

    if error:
        sys.exit()
