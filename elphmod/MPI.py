#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import sys
import numpy as np

try:
    from mpi4py import MPI

except ImportError:
    class Communicator(object):
        def __init__(self):
            self.rank = 0
            self.size = 1

        def Barrier(self):
            pass

        def barrier(self):
            pass

        def Bcast(self, data):
            pass

        def bcast(self, data):
            return data

        def Gatherv(self, send, recv):
            recv[0][...] = send.reshape(recv[0].shape)

        def Allgatherv(self, send, recv):
            recv[0][...] = send.reshape(recv[0].shape)

        def allgather(self, send):
            return [send]

        def Reduce(self, send, recv):
            recv[...] = send.reshape(recv.shape)

        def Allreduce(self, send, recv):
            recv[...] = send.reshape(recv.shape)

        def allreduce(self, send):
            return send

        def Scatterv(self, send, recv):
            recv[...] = send[0].reshape(recv.shape)

        def Split(self, color, key=None):
            return self

        def Split_type(self, color, key=None):
            return self

    class Interface(object):
        def __init__(self):
            self.COMM_WORLD = Communicator()
            self.UNDEFINED = 0
            self.COMM_TYPE_SHARED = self.UNDEFINED

    MPI = Interface()

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

    col = comm.Split(comm.rank % size, key=comm.rank)
    row = comm.Split(comm.rank // size, key=comm.rank) # same col.rank

    return col, row

def shared_array(shape, dtype=float, shared_memory=True, single_memory=False,
        comm=comm):
    """Create array whose memory is shared among all processes on same node.

    With ``shared_memory=False`` (``single_memory=True``) a conventional array
    is created on each (only one) processor, which however allows for the same
    broadcasting syntax as shown below.

    Example:

    .. code-block:: python

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

    .. code-block:: text

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

    if shared_memory and MPI.COMM_TYPE_SHARED == MPI.UNDEFINED:
        # disable shared memory if it is not supported:

        info('Shared memory not implemented')

        shared_memory = False

    if single_memory:
        # pretend that all processors are on same node:

        node = comm # same machine

        if comm.rank == 0:
            array = np.empty(shape, dtype=dtype)
        else:
            array = np.empty(0, dtype=dtype)

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

    # From Gilles reply to Stack Overflow question:
    # 'get Nodes with MPI program in C'

    images = comm.Split(node.rank, key=comm.rank) # same node.rank

    return node, images, array

def load(filename, shared_memory=False, comm=comm):
    """Read and broadcast NumPy data."""

    if comm.rank == 0:
        data = np.load(filename, mmap_mode='r' if shared_memory else None)

        shape = data.shape
        dtype = data.dtype
    else:
        shape = dtype = None

    shape = comm.bcast(shape)
    dtype = comm.bcast(dtype)

    if shared_memory:
        node, images, array = shared_array(shape, dtype=dtype, comm=comm)

        if comm.rank == 0:
            array[...] = data

        if node.rank == 0:
            images.Bcast(array)

        comm.Barrier()

        return array
    else:
        if comm.rank != 0:
            data = np.empty(shape, dtype=dtype)

        comm.Bcast(data)

        return data

def info(message, error=False, comm=comm):
    """Print status message from first process."""

    comm.barrier()

    if comm.rank == 0:
        if error:
            sys.stdout.write('Error: ')

        print(message)

    if error:
        sys.exit()
