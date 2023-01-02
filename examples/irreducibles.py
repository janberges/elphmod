#!/usr/bin/env python3

# Copyright (C) 2017-2023 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod

if elphmod.MPI.comm.rank != 0:
    raise SystemExit

N = 72

points = dict()

for n, m in elphmod.bravais.irreducibles(N, angle=60):
    d = elphmod.bravais.squared_distance(n, m, angle=60)

    if d in points:
        points[d].append((n, m))
    else:
        points[d] = [(n, m)]

print('%4s  %4s  %s' % ('no.', 'r^2', 'irred. points'))

for number, (key, value) in enumerate(sorted(points.items())):
    print('%4d  %4d  %s' % (number, key, ', '.join(map(str, value))))
