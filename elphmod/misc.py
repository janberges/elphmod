#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import numpy as np
import sys

from . import __version__, MPI
comm = MPI.comm
info = MPI.info

verbosity = 1

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

def read_cube(cube):
    """Read Gaussian cube file."""

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
    a = np.empty((3, 3), dtype=float)
    X = np.empty(nat, dtype=int)
    r = np.empty((nat, 3), dtype=float)

    if comm.rank == 0:
        for i in range(3):
            cols = next(lines).split()
            n[i] = int(cols[0])
            a[i] = list(map(float, cols[1:4]))

        for i in range(nat):
            cols = next(lines).split()
            X[i] = int(cols[0])
            r[i] = list(map(float, cols[1:4]))

    comm.Bcast(n)
    comm.Bcast(a)
    comm.Bcast(X)
    comm.Bcast(r)

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

    return a, X, r, data

def read_xsf(xsf):
    """Read file in XCrySDen format."""

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
    r = np.empty((nat, 3), dtype=float)

    if comm.rank == 0:
        for i in range(nat):
            cols = next(lines).split()
            X[i] = cols[0]
            r[i] = list(map(float, cols[1:4]))

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
    comm.Bcast(r)

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

    return r0, a, X, r, data

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
