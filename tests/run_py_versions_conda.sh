#!/bin/bash

# conda install conda-build
# source ./conda.sh

env=/dev/shm/env
log=run_py_versions_conda.log

echo "Tests for different Python versions" > $log

for minor in `seq 5 13`
do
    conda create -y -p $env python=3.$minor
    conda activate $env

    conda install -y numpy scipy
    conda develop ..

    echo "Tests for Python 3.$minor" | tee -a $log
    python3 -m unittest -vfc 2>&1 | tee -a $log

    conda deactivate
    conda remove -y -p $env --all
done
