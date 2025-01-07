#!/bin/bash

# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

eval `elphmodenv`

set -e

for example in *.py
do
    echo $example
    mpirun python3 $example
done

for example in */run.sh
do
    pushd `dirname $example`
    ./run.sh
    popd
done
