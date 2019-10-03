#/usr/bin/env python

import sys
import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD

def distribute(size, bounds=False, comm=comm, chunks=None):
    """Distribute work among processes."""

    if chunks is None:
        chunks = comm.size

    sizes = np.empty(chunks, dtype=int)

    if comm.rank == 0:
        sizes[:] = size // chunks
        sizes[:size % chunks] += 1

    comm.Bcast(sizes)

    if bounds:
        cumsum = np.empty(chunks + 1, dtype=int)

        if comm.rank == 0:
            cumsum[0] = 0

            for rank in range(chunks):
                cumsum[rank + 1] = cumsum[rank] + sizes[rank]

        comm.Bcast(cumsum)

        return sizes, cumsum

    return sizes

def matrix(size, comm=comm):
    """Create sub-communicators."""

    sizes, cumsum = distribute(size, bounds=True, comm=comm)

    col = comm.Split(comm.rank %  size, key=comm.rank)
    row = comm.Split(comm.rank // size, key=comm.rank) # same col.rank

    return col, row

def shared_array(shape, dtype=float, shared_memory=True, single_memory=False,
        comm=comm):
    """Create array whose memory is shared among all processes on same node.

    With ``shared_memory=False`` (``single_memory=True``) a conventional array
    is created on each (only one) processor, which however allows for the same
    broadcasting syntax as shown below.

    Example:

    # Set up huge array:
    node, images, array = shared_array(2 ** 30, dtype=np.uint8)

    # Write data on one node:
    if comm.rank == 0:
        array[:] = 0

    # Broadcast data to other nodes:
    if node.rank == 0:
        images.Bcast(array)

    # Wait if node.rank != 0:
    comm.Barrier()
                               ______ ______
                    Figure:   |_0,_0_|_4,_0_|
                              |_1,_1_|_5,_1_|
    comm.rank and node.rank   |_2,_2_|_6,_2_|
    on machine with 2 nodes   |_3,_3_|_7,_3_|
    with 4 processors each.    node 1 node 2

    Because of the sorting ``key=comm.rank`` in the split functions below,
    ``comm.rank == 0`` is equivalent to ``node.rank == images.rank == 0``.
    """
    dtype = np.dtype(dtype)

    if MPI.COMM_TYPE_SHARED == MPI.UNDEFINED:
        # workaround if shared memory is not supported:

        info("Shared memory not implemented")

        shared_memory = False

    if single_memory:
        # pretend that all processors are on same node:

        node = comm # same machine

        if comm.rank == 0:
            array = np.empty(shape, dtype=dtype)
        else:
            array = None

    elif shared_memory:
        # From article from Intel Developer Zone:
        # 'An Introduction to MPI-3 Shared Memory Programming'

        node = comm.Split_type(MPI.COMM_TYPE_SHARED, key=comm.rank) # same node

        # Shared memory allocation following Lisandro Dalcin on Google Groups:
        # 'Shared memory for data structures and mpi4py.MPI.Win.Allocate_shared'

        size = np.prod(shape) * dtype.itemsize if node.rank == 0 else 0

        window = MPI.Win.Allocate_shared(size, dtype.itemsize, comm=node)
        buffer, itemsize = window.Shared_query(0)

        array = np.ndarray(shape, buffer=buffer, dtype=dtype)

    else:
        # pretend that each processor is on separate node:

        node = comm.Split(comm.rank) # same core

        array = np.empty(shape, dtype=dtype)

    # From Gilles reply to StackOverflow question:
    # 'get Nodes with MPI program in C'

    images = comm.Split(node.rank, key=comm.rank) # same node.rank

    return node, images, array

def info(message, error=False, comm=comm):
    """Print status message from first process."""

    comm.barrier()

    if comm.rank == 0:
        if error:
            sys.stdout.write('Error: ')

        print(message)

    if error:
        sys.exit()
