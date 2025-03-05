# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

"""Work distribution and shared memory."""

import numpy as np
import sys

try:
    from mpi4py import MPI

    if MPI.COMM_WORLD.size == 1:
        raise ImportError

except ImportError:
    class Communicator:
        def __init__(self):
            self.rank = 0
            self.size = 1

        @staticmethod
        def Barrier():
            pass

        @staticmethod
        def barrier():
            pass

        @staticmethod
        def Bcast(data):
            pass

        @staticmethod
        def bcast(data):
            return data

        @staticmethod
        def Gatherv(send, recv):
            recv[0][...] = send.reshape(recv[0].shape)

        @staticmethod
        def Allgatherv(send, recv):
            recv[0][...] = send.reshape(recv[0].shape)

        @staticmethod
        def gather(send):
            return [send]

        @staticmethod
        def allgather(send):
            return [send]

        @staticmethod
        def Reduce(send, recv):
            if send is not MPI.IN_PLACE:
                recv[...] = send.reshape(recv.shape)

        @staticmethod
        def Allreduce(send, recv):
            if send is not MPI.IN_PLACE:
                recv[...] = send.reshape(recv.shape)

        @staticmethod
        def allreduce(send):
            return send

        @staticmethod
        def Scatterv(send, recv):
            recv[...] = send[0].reshape(recv.shape)

        def Split(self, color, key=None):
            return self

        def Split_type(self, color, key=None):
            return self

    class Interface:
        def __init__(self):
            self.COMM_WORLD = Communicator()
            self.UNDEFINED = 0
            self.COMM_TYPE_SHARED = self.UNDEFINED
            self.IN_PLACE = 1

    MPI = Interface()

comm = MPI.COMM_WORLD

I = comm.Split(comm.rank)

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

def shm_split(comm=comm, shared_memory=True):
    """Create communicators for use with shared memory.

    Parameters
    ----------
    comm : MPI.Intracomm
        Overarching communicator.
    shared_memory : bool, default True
        Use shared memory? Provided for convenience.

    Returns
    -------
    node : MPI.Intracomm
        Communicator between processes that share memory (on the same node).
    images : MPI.Intracomm
        Communicator between processes that have the same ``node.rank``.

    Warnings
    --------
    If shared memory is not implemented, each process shares memory only with
    itself. A warning is issued.

    Notes
    -----
    Visualization for a machine with 2 nodes with 4 processors each::

         ________________ ________________ ________________ ________________
        | comm.rank: 0   | comm.rank: 1   | comm.rank: 2   | comm.rank: 3   |
        | node.rank: 0   | node.rank: 1   | node.rank: 2   | node.rank: 3   |
        | images.rank: 0 | images.rank: 0 | images.rank: 0 | images.rank: 0 |
        |________________|________________|________________|________________|
         ________________ ________________ ________________ ________________
        | comm.rank: 4   | comm.rank: 5   | comm.rank: 6   | comm.rank: 7   |
        | node.rank: 0   | node.rank: 1   | node.rank: 2   | node.rank: 3   |
        | images.rank: 1 | images.rank: 1 | images.rank: 1 | images.rank: 1 |
        |________________|________________|________________|________________|

    Since both ``node.rank`` and ``images.rank`` are sorted by ``comm.rank``,
    ``comm.rank == 0`` is equivalent to ``node.rank == images.rank == 0``.
    """
    if not shared_memory or MPI.COMM_TYPE_SHARED == MPI.UNDEFINED:
        if shared_memory and comm.size > 1:
            info('Shared memory not implemented')

        return I, comm

    # From article from Intel Developer Zone:
    # 'An Introduction to MPI-3 Shared Memory Programming'

    node = comm.Split_type(MPI.COMM_TYPE_SHARED, key=comm.rank)

    # From Gilles reply to Stack Overflow question:
    # 'get Nodes with MPI program in C'

    images = comm.Split(node.rank, key=comm.rank)

    return node, images

def shared_array(shape, dtype=float, shared_memory=True, single_memory=False,
        only_info=False, comm=comm):
    """Create array whose memory is shared among all processes on same node.

    With ``shared_memory=False`` (``single_memory=True``) a conventional array
    is created on each (only one) processor, which however allows for the same
    broadcasting syntax as shown below.

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
    """
    buffer = None
    dtype = np.dtype(dtype)

    if single_memory:
        node, images = comm, I

        if comm.rank != 0:
            shape = 0
    else:
        node, images = shm_split(comm, shared_memory)

        # Shared memory allocation following Lisandro Dalcin on Google Groups:
        # 'Shared memory for data structures and mpi4py.MPI.Win.Allocate_shared'

        if node.size > 1:
            if node.rank == 0:
                size = np.prod(shape, dtype=int) * dtype.itemsize
            else:
                size = 0

            window = MPI.Win.Allocate_shared(size, dtype.itemsize, comm=node)
            buffer, itemsize = window.Shared_query(0)

            if comm.rank == 0:
                if size > 1e9:
                    print('About to allocate %.6f GB of shared memory'
                        % (size / 1e9))

    if only_info:
        return node, images, shape, buffer, dtype
    else:
        return node, images, np.ndarray(shape, buffer=buffer, dtype=dtype)

class SharedArray(np.ndarray):
    """Create array whose memory is shared among all processes on same node.

    See Also
    --------
    shared_array
    """
    def __new__(cls, *args, **kwargs):
        node, images, shape, buffer, dtype = shared_array(only_info=True,
            *args, **kwargs)

        array = super(SharedArray, cls).__new__(cls, shape,
            buffer=buffer, dtype=dtype)

        array.node = node
        array.images = images

        return array

    def __array_finalize__(self, array):
        if array is not None:
            self.node = getattr(array, 'node', I)
            self.images = getattr(array, 'images', comm)

    def Bcast(self):
        if self.node.rank == 0:
            self.images.Bcast(self)

        comm.Barrier()

class Buffer:
    """Wrapper for ``pickle`` in parallel context.

    Parameters
    ----------
    buf : str, default None
        Name of buffer file with pickled object. By default, the buffer is not
        used and :meth:`get` and :meth:`set` do nothing and return ``None``.
    """
    def __init__(self, buf=None):
        global pickle
        import pickle

        self.buf = buf

    def get(self):
        """Read object from buffer and broadcast it.

        Returns
        -------
        object
            Object if buffer exists, ``None`` otherwise.
        """
        if self.buf is None:
            return

        obj = None

        if comm.rank == 0:
            try:
                with open(self.buf, 'rb') as data:
                    obj = pickle.load(data)
            except FileNotFoundError:
                obj = None

        return comm.bcast(obj)

    def set(self, obj):
        """Write object to buffer on single processor.

        Parameters
        ----------
        object
            Object to write.
        """
        if self.buf is None:
            return

        if comm.rank == 0:
            with open(self.buf, 'wb') as data:
                pickle.dump(obj, data, protocol=4)

def load(filename, shared_memory=False, txt=False, comm=comm):
    """Read and broadcast NumPy data.

    Parameters
    ----------
    shared_memory : bool, default False
        Use shared memory?
    txt : bool, default False
        Load data from text file?

    Returns
    -------
    ndarray
        Data.
    """
    if comm.rank == 0:
        try:
            if txt:
                data = np.loadtxt(filename)
            else:
                data = np.load(filename,
                    mmap_mode='r' if shared_memory else None)

        except IOError:
            data = np.empty(0)

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

        sys.stdout.flush()

    if error:
        sys.exit()

def elphmodenv(num_threads=1):
    """Print commands to change number of threads.

    Run `eval $(elphmodenv)` before your Python scripts to prevent NumPy from
    using multiple processors for linear algebra. This is advantageous when
    parallelizing with MPI already (`mpirun python3 script.py`) or on shared
    servers, where CPU resources must be used with restraint.

    Combine `eval $(elphmodenv X)` with `mpirun -n Y python3 script.py`, where
    the product of X an Y is the number of available CPUs, for hybrid
    parallelization.
    """
    if len(sys.argv) > 1:
        num_threads = int(sys.argv[1])

    # From MehmedB @ Stack Overflow: "How to limit number of CPU's used by a
    # python script w/o terminal or multiprocessing library?"

    for var in ['MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        info('export %s=%d' % (var, num_threads))
