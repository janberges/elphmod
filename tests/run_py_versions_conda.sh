#!/bin/bash

# conda install conda-build
# source run_py_versions_conda.sh

env=/dev/shm/env
log=run_py_versions_conda.log

#unset env # use existing environments py3.5, py3.6, etc.

echo "Tests for different Python versions" > $log

for minor in `seq 5 14`
do
    if test $env
    then
        conda create -y -p $env python=3.$minor
        conda activate $env

        conda install -y numpy scipy
        conda develop ..
    else
        conda activate py3.$minor
    fi

    echo "Tests for Python 3.$minor" | tee -a $log
    python3 -m unittest -vfc 2>&1 | tee -a $log

    conda deactivate

    if test $env
    then
        conda remove -y -p $env --all
    fi
done
