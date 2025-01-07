#!/usr/bin/env python3

# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import numpy as np
import storylines
import sys

black = np.array([0x00, 0x00, 0x00])
white = np.array([0xff, 0xff, 0xff])
blue = np.array([0x19, 0x3b, 0xda])
red = np.array([0xda, 0x19, 0x3b])

N = 13

shades = (
    np.linspace(white, black, N),
    np.linspace(white, blue, N),
    np.linspace(white, red, N),
    np.linspace(black, red, N))

image = np.array(storylines.load(sys.argv[1]))[:, :, :3]

for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        for shade in shades:
            n = shade[-1] - shade[0]
            n /= np.linalg.norm(n)

            d = image[y, x] - shade[0]

            if np.linalg.norm(d - np.dot(d, n) * n) < 1:
                image[y, x] = min(shade,
                    key=lambda c: np.linalg.norm(c - image[y, x]))
                break
        else:
            print('Original color (%d, %d, %d) is kept.' % tuple(image[y, x]))

storylines.save(sys.argv[2], image)
