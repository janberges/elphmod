#!/bin/bash

set -e

echo "Serial tests"
python3 -m unittest -vfc

echo "Parallel tests"
mpirun -n 2 python3 -m unittest -fc
