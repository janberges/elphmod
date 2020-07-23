#/usr/bin/env python3

import elphmod

if elphmod.MPI.comm.rank != 0:
    raise SystemExit

N = 72

points = dict()

for n, m in elphmod.bravais.irreducibles(N, angle=60):
    d = n * n + n * m + m * m

    if d in points:
        points[d].append((n, m))
    else:
        points[d] = [(n, m)]

print('%4s  %4s  %s' % ('no.', 'r^2', 'irred. points'))

for number, (key, value) in enumerate(sorted(points.items())):
    print('%4d  %4d  %s' % (number, key, ', '.join(map(str, value))))
