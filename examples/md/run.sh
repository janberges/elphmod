#!/bin/bash

# Copyright (C) 2017-2023 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

cp $IPI_ROOT/drivers/py/driver.py .

python3 driver.py --unix
