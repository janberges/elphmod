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
    """Create array whose memory is shared among all processes on same node.

    Example:

    # Set up huge array:
    node, images, array = shared_array(2 ** 30, dtype=np.uint8)

    # Write data on one node:
    if comm.rank == 0:
        array[:] = 0

    # Broadcast data to other nodes:
    if node.rank == 0:
        images.Bcast(array)
                               ______ ______
                    Figure:   |_0,_0_|_4,_0_|
                              |_1,_1_|_5,_1_|
    comm.rank and node.rank   |_2,_2_|_6,_2_|
    on machine with 2 nodes   |_3,_3_|_7,_3_|
    with 4 processors each.    node 1 node 2
    """

    # From article from Intel Developer Zone:
    # 'An Introduction to MPI-3 Shared Memory Programming'

    node = comm.Split_type(MPI.COMM_TYPE_SHARED, key=comm.rank) # same node

    # From Gilles reply to StackOverflow question:
    # 'get Nodes with MPI program in C'

    images = comm.Split(node.rank, key=comm.rank) # same node.rank

    # Shared memory allocation following Lisandro Dalcin on Google Groups:
    # 'Shared memory for data structures and mpi4py.MPI.Win.Allocate_shared'

    size = np.prod(shape)
    dtype = np.dtype(dtype)
    itemsize = dtype.itemsize

    if node.rank == 0:
        bytes = size * itemsize
    else:
        bytes = 0

    win = MPI.Win.Allocate_shared(bytes, itemsize, comm=node)
    buf, itemsize = win.Shared_query(0)

    return node, images, np.ndarray(shape, buffer=buf, dtype=dtype)

def collect(my_data, shape, sizes, dtype, shared_memory=True):
    """Gather data of variable sizes into shared memory."""

    elements = sizes * np.prod(shape) // np.sum(sizes)

    if shared_memory:
        node, images, data = shared_array(shape, dtype=dtype)

        comm.Gatherv(my_data, (data, elements))

        if node.rank == 0:
            images.Bcast(data)
    else:
        data = np.empty(shape, dtype=dtype)

        comm.Allgatherv(my_data, (data, elements))

    return data

def info(message, error=False):
    """Print status message from first process."""

    comm.barrier()

    if comm.rank == 0:
        if error:
            sys.stdout.write('Error: ')

        print(message)

    if error:
        sys.exit()
