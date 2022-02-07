#!/usr/bin/env python3

# Copyright (C) 2017-2022 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import numpy as np
import sys

from . import __version__, MPI
comm = MPI.comm
info = MPI.info

verbosity = 1
# 0: suppress all output
# 1: only print warnings
# 2: print info messages
# 3: display status bars

# constants in SI units:
cSI = 299792458.0 # speed of light (m/s)
eVSI = 1.602176634e-19 # electronvolt (J)
hSI = 6.62607015e-34 # Planck constant (J s)
hbarSI = hSI / (2.0 * np.pi) # reduced Planck constant (J s)
kBSI = 1.380649e-23 # Boltzmann constant (J/K)
NA = 6.02214076e23 # Avogadro constant (1/mol)
uSI = 1e-3 / NA # atomic mass constant (kg)

# constants in atomic units (eV, AA):
Ry = 13.605693122994 # Rydberg energy (eV) [1]
a0 = 0.529177210903 # Bohr radius (AA) [1]
cmm1 = 100.0 * hSI * cSI / eVSI # "inverse cm" (eV)
kB = kBSI / eVSI # Boltzmann constant (eV/K)

# [1] 2018 CODATA recommended values

class StatusBar(object):
    def __init__(self, count, width=60, title='progress'):
        if comm.rank:
            return

        self.counter = 0
        self.count = count
        self.width = width
        self.progress = 0

        sys.stdout.write((' %s ' % title).center(width, '_'))
        sys.stdout.write('\n')

    def update(self):
        if comm.rank:
            return

        self.counter += 1

        progress = self.width * self.counter // self.count

        if progress != self.progress:
            sys.stdout.write('=' * (progress - self.progress))
            sys.stdout.flush()

            self.progress = progress

        if self.counter == self.count:
            sys.stdout.write('\n')

def hello():
    if verbosity:
        info('This is elphmod (version %s) running on %d processors.'
            % (__version__, comm.size))
        info('To suppress all output, set elphmod.misc.verbosity = 0.')

def group(points, eps=1e-7):
    """Group points into neighborhoods.

    Parameters
    ----------
    points : ndarray
        Points to be grouped.
    eps : float
        Maximal distance between points in the same group.

    Returns
    -------
    list of lists
        Groups of indices.
    """
    groups = np.arange(len(points))

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if np.all(np.absolute(points[j] - points[i]) < eps):
                groups[np.where(groups == groups[j])] = groups[i]

    return [np.where(groups == group)[0] for group in set(groups)]

def read_cube(cube, only_header=False, comm=comm):
    """Read Gaussian cube file.

    Parameters
    ----------
    cube : str
        Name of Gaussian cube file.
    only_header : bool, default False
        Skip reading data?

    Returns
    -------
    r0 : ndarray
        Origin of the data grid.
    a : ndarray
        Spanning vectors of the data grid.
    X : list of str
        Atomic numbers.
    tau : ndarray
        Cartesian atomic coordinates.
    data : ndarray
        Data-grid values or, if `only_header`, shape of data grid.

    See Also
    --------
    read_xsf : Equivalent function for XCrySDen format.
    """
    if comm.rank == 0:
        lines = open(cube)

        next(lines)
        next(lines)

        cols = next(lines).split()
        nat = int(cols[0])
        r0 = np.array(list(map(float, cols[1:4])))
    else:
        nat = None

    nat = comm.bcast(nat)

    n = np.empty(3, dtype=int)
    r0 = np.empty(3, dtype=float)
    a = np.empty((3, 3), dtype=float)
    X = np.empty(nat, dtype=int)
    tau = np.empty((nat, 3), dtype=float)

    if comm.rank == 0:
        for i in range(3):
            cols = next(lines).split()
            n[i] = int(cols[0])
            a[i] = list(map(float, cols[1:4]))
            a[i] *= n[i]

        for i in range(nat):
            cols = next(lines).split()
            X[i] = int(cols[0])
            tau[i] = list(map(float, cols[2:5]))

    comm.Bcast(n)
    comm.Bcast(r0)
    comm.Bcast(a)
    comm.Bcast(X)
    comm.Bcast(tau)

    if only_header:
        return r0, a, X, tau, tuple(n)

    if comm.rank == 0:
        data = np.empty(np.prod(n))

        i = 0
        for line in lines:
            for number in line.split():
                data[i] = float(number)
                i += 1

        data = np.reshape(data, n)

        lines.close()
    else:
        data = np.empty(n, dtype=float)

    comm.Bcast(data)

    return r0, a, X, tau, data

def read_xsf(xsf, only_header=False, comm=comm):
    """Read file in XCrySDen format.

    Parameters
    ----------
    xsf : str
        Name of XCrySDen file.
    only_header : bool, default False
        Skip reading data?

    Returns
    -------
    r0 : ndarray
        Origin of the data grid.
    a : ndarray
        Spanning vectors of the data grid.
    X : list of str
        Atomic numbers or symbols.
    tau : ndarray
        Cartesian atomic coordinates.
    data : ndarray
        Data-grid values or, if `only_header`, shape of data grid.

    See Also
    --------
    read_cube : Equivalent function for Gaussian cube format.
    write_xsf : Write file in XCrySDen format.
    """
    if comm.rank == 0:
        lines = open(xsf)

        while next(lines).strip() != 'PRIMCOORD':
            pass

        nat = int(next(lines).split()[0])
    else:
        nat = None

    nat = comm.bcast(nat)

    n = np.empty(3, dtype=int)
    r0 = np.empty(3, dtype=float)
    a = np.empty((3, 3), dtype=float)
    X = ['X'] * nat
    tau = np.empty((nat, 3), dtype=float)

    if comm.rank == 0:
        for i in range(nat):
            cols = next(lines).split()
            X[i] = cols[0]
            tau[i] = list(map(float, cols[1:4]))

        while next(lines).strip() != 'BEGIN_DATAGRID_3D_UNKNOWN':
            pass

        n[:] = list(map(int, next(lines).split()))
        r0[:] = list(map(float, next(lines).split()))

        for i in range(3):
            a[i] = list(map(float, next(lines).split()))

    comm.Bcast(n)
    comm.Bcast(r0)
    comm.Bcast(a)
    X = comm.bcast(X)
    comm.Bcast(tau)

    if only_header:
        return r0, a, X, tau, tuple(n)

    if comm.rank == 0:
        data = np.empty(np.prod(n))

        i = 0
        while i < data.size:
            for number in next(lines).split():
                data[i] = float(number)
                i += 1

        data = np.reshape(data, n[::-1])
        data = np.transpose(data).copy()

        lines.close()
    else:
        data = np.empty(n, dtype=float)

    comm.Bcast(data)

    return r0, a, X, tau, data

def write_xsf(xsf, r0, a, X, tau, data, only_header=False, comm=comm):
    """Write file in XCrySDen format.

    Parameters
    ----------
    xsf : str
        Name of XCrySDen file.
    r0 : ndarray
        Origin of the data grid.
    a : ndarray
        Spanning vectors of the data grid.
    X : list of str
        Atomic numbers or symbols.
    tau : ndarray
        Cartesian atomic coordinates.
    data : ndarray
        Data-grid values or, if `only_header`, shape of data grid.
    only_header : bool, default False
        Skip writing data?

    See Also
    --------
    read_xsf : Read file in XCrySDen format.
    """
    if comm.rank == 0:
        n = tuple(data) if only_header else data.shape

        with open(xsf, 'w') as text:
            text.write('PRIMCOORD\n')
            text.write('%6d  1\n' % len(X))

            for i in range(len(X)):
                text.write('%-5s' % X[i])
                text.write(' %11.7f %11.7f %11.7f\n' % tuple(tau[i]))

            text.write('BEGIN_BLOCK_DATAGRID_3D\n')
            text.write('3D_field\n')
            text.write('BEGIN_DATAGRID_3D_UNKNOWN\n')
            text.write('%6d %5d %5d\n' % tuple(n))
            text.write('%12.6f %11.6f %11.6f\n' % tuple(r0))

            for i in range(3):
                text.write('%12.7f %11.7f %11.7f\n' % tuple(a[i]))

            if not only_header:
                for i, value in enumerate(data.flat, 1):
                    text.write('%13.5e' % value)

                    if not i % 6 or i == data.size:
                        text.write('\n')

            text.write('END_DATAGRID_3D\n')
            text.write('END_BLOCK_DATAGRID_3D\n')

def real_space_grid(shape, r0, a, shared_memory=False):
    """Sample real-space grid.

    Parameters
    ----------
    shape : tuple of int
        Shape of the 3D real-space grid.
    r0 : ndarray
        Origin of the real-space grid.
    a : ndarray
        Cartesian spanning vectors of the real-space grid.
    shared_memory : bool, default False
        Store real-space grid in shared memory?

    Returns
    -------
    ndarray
        Cartesian coordinates for all grid points.
    """
    node, images, r = MPI.shared_array(shape + (3,),
        shared_memory=shared_memory)

    if comm.rank == 0:
        axes = [np.linspace(0.0, 1.0, num) for num in shape]
        axes = np.meshgrid(*axes, indexing='ij')
        r[...] = r0 + np.einsum('nijk,nx->ijkx', axes, a)

    if node.rank == 0:
        images.Bcast(r)

    comm.Barrier()

    return r

def read_namelists(filename):
    """Extract all Fortran namelists from file.

    Parameters
    ----------
    filename : str
        Name of file with namelists.

    Returns
    -------
    dict of dict
        Namelist data.
    """
    import re

    with open(filename) as data:
        text = data.read()

    # remove comments:

    text = re.sub(r'!.*$', '', text, flags=re.MULTILINE)

    # protect string (which might contain spaces):

    groups = []

    def replace(match):
        groups.append(match.group(0))
        return '___%d___' % len(groups)

    def place(match):
        return groups[int(match.group(0).strip('_')) - 1]

    count = 1
    while count:
        text, count = re.subn(r'\'[^\'"]*\'|"[^\'"]*"', replace, text)

    # parse all namelists:

    data = dict()

    for namelist in re.finditer(r'&(\w+)(.+?)/', text, flags=re.DOTALL):
        name, content = namelist.groups()
        name = name.lower()
        data[name] = dict()

        content = re.sub(r'=', ' = ', content)
        content = re.sub(r'\s+\(', '(', content)
        items = re.split(r'[;,\s]+', content)

        key = None
        for i, item in enumerate(items):
            if not item:
                continue

            if '=' in items[i + 1:i + 2]:
                key, n = re.match(r'([^(]*)[( ]*(\d*)', item.lower()).groups()
                n = int(n) - 1 if n else 0

                if not key in data[name]:
                    data[name][key] = []

                while len(data[name][key]) < n:
                    data[name][key].append(None)

            elif key is not None and item != '=':
                count = 1
                while count:
                    item, count = re.subn('___\d+___', place, item)

                if re.match(r'(\'.*\'|".*")$', item):
                    item = item[1:-1]
                elif re.search('[Tt]', item):
                    item = True
                elif re.search('[Ff]', item):
                    item = False
                else:
                    try:
                        item = int(item)
                    except ValueError:
                        try:
                            item = float(re.sub(r'[dDE]', 'e', item))
                        except ValueError:
                            pass

                data[name][key][n:n + 1] = [item]
                n += 1

        for key, value in data[name].items():
            if len(value) == 1:
                data[name][key] = value[0]

    return data

def read_input_data(filename, broadcast=True):
    """Read Quantum ESPRESSO input data.

    Parameters
    ----------
    filename : str
        Name of input file.
    broadcast : bool
        Broadcast result from rank 0 to all processes?

    Returns
    -------
    dict
        Input data.
    """
    if comm.rank == 0:
        struct = dict()

        for namelist in read_namelists(filename).values():
            struct.update(namelist)
    else:
        struct = None

    if broadcast:
        struct = comm.bcast(struct)

    return struct

def split(expr, sd=',', od='{', cd='}'):
    """Split expression with separators and brackets using distributive law.

    Parameters
    ----------
    expr : str
        Expression to be expanded and split.
    sd : str
        Separating delimiter.
    od : str
        Opening delimiter.
    cd : str
        Closing delimiter.

    Returns
    -------
    generator
        Separated elements of expression.
    """
    import re

    group = '[{0}]([^{0}{1}]*)[{1}]'.format(*
        [re.sub(r'([\^\]])', r'\\\1', d) for d in (od, cd)])

    groups = []

    def pack(match):
        groups.append(match.group(1))
        return '<%d>' % (len(groups) - 1)

    expr = od + expr + cd

    n = 1
    while n:
        expr, n = re.subn(group, pack, expr)

    def factorize(x):
        match = re.match(r'(.*)<(\d+)>(.*)', x)

        if match:
            for y in groups[int(match.group(2))].split(sd):
                for z in factorize(match.group(1) + y.strip() + match.group(3)):
                    yield z
        else:
            yield x

    return factorize(expr)

def vector_index(vectors, vector):
    """Find index of vector in list of vectors.

    Parameters
    ----------
    vectors : ndarray
        List of vectors.
    vector : ndarray
        Vector.

    Returns
    -------
    int
        Index of vector in list of vectors.
    """
    match = np.all(vectors == vector, axis=1)

    if np.any(match):
        return np.argmax(match)
    else:
        return None
