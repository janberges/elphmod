#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import numpy as np
import sys

from . import MPI
comm = MPI.comm

Ry = 13.605693009 # Rydberg energy (eV)
a0 = 0.52917721090380 # Bohr radius (AA)
kB = 8.61733e-5 # Boltzmann constant (eV/K)

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
