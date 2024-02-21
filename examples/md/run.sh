#!/bin/bash

# Copyright (C) 2017-2024 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

if test -n "$IPI_ROOT"
then
    cp $IPI_ROOT/drivers/py/driver.py .
else
    wget https://raw.githubusercontent.com/i-pi/i-pi/master/drivers/py/driver.py
fi

python3 driver.py --unix
