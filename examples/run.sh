#!/bin/bash

# Copyright (C) 2017-2026 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

eval `elphmodenv`

set -e

: ${NP:=2}
: ${NK:=2}

for example in *.py
do
    echo $example
    mpirun -n $NP python3 $example
done

export NP
export NK

for example in */run.sh
do
    pushd `dirname $example`
    ./run.sh
    popd
done
