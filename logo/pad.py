#!/usr/bin/env python3

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import numpy as np
import storylines

def pad(i, ho, wo):
    hi, wi, c = i.shape

    o = np.full((ho, wo, c), 255)
    o[(ho - hi) // 2:(ho + hi) // 2, (wo - wi) // 2:(wo + wi) // 2] = i

    return o

image = np.array(storylines.load('elphmod.png'))

storylines.save('elphmod_square.png', pad(image, 480, 480))
storylines.save('elphmod_banner.png', pad(image, 640, 1280))
