elphmod
=======

This is a collection of Python modules to handle electron-phonon models:

.. toctree::
   :maxdepth: 2

   modules/el
   modules/ph
   modules/elph
   modules/elel
   modules/MPI
   modules/bravais
   modules/dispersion
   modules/dos
   modules/diagrams
   modules/occupations
   modules/eliashberg
   modules/plot
   modules/misc

Installation
============

elphmod depends on MPI (with shared-memory support). Using APT, it can be
installed as follows:

.. code-block:: bash

    sudo apt install libopenmpi-dev

It also depends on the Python packages listed in the file "requirements.txt".
One way to install them is inside a virtual environment:

.. code-block:: bash

    python3 -m venv elphmod.venv
    source elphmod.venv/bin/activate
    pip3 install -r requirements.txt

Documentation
=============

All functions are documented directly in the source files using NumPy-style
docstrings. You can generate an automatic documentation in HTML format using
Sphinx:

.. code-block:: bash

    cd doc
    make html
    firefox html/index.html

Please also have a look at the examples:

.. code-block:: bash

    cd examples
    mpirun python3 electrons.py
    mpirun python3 phonons.py

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
