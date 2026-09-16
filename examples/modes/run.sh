#!/bin/bash

# Copyright (C) 2017-2026 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

: ${NP:=2}

mpirun -n $NP python3 modes.py
