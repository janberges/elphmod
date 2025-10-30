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
        readme=../examples/$name/README.md

        if ! test -f $rst
        then
            if test -f $readme
            then
                echo ".. include:: ../$readme" >> $rst
                echo "   :parser: myst_parser.sphinx_" >> $rst
            else
                echo $name > $rst
                echo $name | sed 's/./=/g' >> $rst
            fi

            echo >> $rst
            echo ".. literalinclude:: ../$example" >> $rst
            echo "   :language: python" >> $rst
            echo >> $rst

            for figure in $folder/$name*.png
            do
                echo ".. image:: ../$figure" >> $rst
            done
        fi

        if test -f $readme
        then
            title=`head -n 1 $readme`
            printf "%-40s %s\n" "${title#\# }" examples/$name
        else
            printf "%-40s %s\n" "`head -n 1 $rst`" examples/$name
        fi
    fi
done
