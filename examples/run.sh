#!/bin/bash

# Copyright (C) 2017-2022 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

set -e

for example in *.py
do
    echo $example
    mpirun python3 $example
done

for example in \
    goldstone \
    lambda \
    modes \
    phrenorm \
    phrenorm_3d \
    phrenorm_graphene \
    projwfc projwfc_3d \
    simstm \
    simsts \
    wannier
do
    pushd $example
    ./run.sh
    popd
done
