#!/bin/bash

mkdir -p examples

for example in ../examples/*.py ../examples/*/*.py
do
    folder=`dirname $example`
    name=`basename $example`
    name=${name%.py}

    if ls $folder/$name*.png > /dev/null 2>&1
    then
        rst=examples/$name.rst

        if ! test -f $rst
        then
            echo $name > $rst
            echo $name | sed 's/./=/g' >> $rst
            echo >> $rst
            echo ".. literalinclude:: ../$example" >> $rst
            echo "   :language: python" >> $rst
            echo >> $rst

            for figure in $folder/$name*.png
            do
                echo ".. image:: ../$figure" >> $rst
            done
        fi

        printf "%-40s %s\n" "`head -n 1 $rst`" examples/$name
    fi
done
