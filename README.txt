~ e l p h m o d ~

Python modules for electron-phonon models


I n s t a l l a t i o n

elphmod depends on the Python packages listed in the file "requirements.txt".
One way to install them is inside a virtual environment:

    $ python3 -m venv elphmod.venv
    $ source elphmod.venv/bin/activate
    $ pip3 install -r triqs.src/requirements.txt


E x a m p l e

    $ cd examples
    $ mpirun -np 4 python3 -u example.py

will show you:

    (1) comparison with phonon dispersion from Quantum ESPRESSO's matdyn.x
    (2) ordered phonon bands along winding path through reciprocal unit cell
    (3) Fermi-surface average of electron-phonon interaction for each band
    (4) phonon density of states and shape of Eliashberg spectral function
