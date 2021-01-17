#!/bin/bash

# Based on script pp.py provided with EPW code (C) 2015 Samuel Ponce.
# This program is free software under the terms of the GNU GPLv3 or later.

prefix=TaS2
points=2

mkdir -p save

cp -r work/_ph0/$prefix.phsave save

for n in `seq 1 $points`
do
    cp dyn$n save/$prefix.dyn_q$n

    path=work/_ph0

    test $n -gt 1 && path=$path/$prefix.q_$n

    cp $path/$prefix.dvscf1 save/$prefix.dvscf_q$n
done
