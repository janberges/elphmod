#!/usr/bin/env python3

# Copyright (C) 2017-2022 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import numpy as np

if elphmod.MPI.comm.rank != 0:
    raise SystemExit

width = 400
height = 300

for colors in 1, 3, 4:
    filename = 'image_io_%d.png' % colors

    before = np.random.randint(0, 255, (height, width, colors), dtype=np.uint8)

    elphmod.plot.save(filename, before)

    after = elphmod.plot.load(filename)

    if np.all(before == after):
        print("The image '%s' has been correclty written and read." % filename)
