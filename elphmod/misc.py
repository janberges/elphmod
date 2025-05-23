# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

"""Constants, status bars, parsing, etc."""

import collections
import numpy as np
import sys

import elphmod.MPI

comm = elphmod.MPI.comm
info = elphmod.MPI.info

verbosity = 3
"""Level of verbosity.

- ``0``: Suppress all output.
- ``1``: Only print warnings.
- ``2``: Print info messages.
- ``3``: Display status bars.
"""

# exact constants in SI units:

cSI = 299792458.0
"""Speed of light (m/s)."""

eVSI = 1.602176634e-19
"""Electronvolt (J)."""

hSI = 6.62607015e-34
"""Planck constant (J s)."""

hbarSI = hSI / (2.0 * np.pi)
"""reduced Planck constant (J s)."""

kBSI = 1.380649e-23
"""Boltzmann constant (J/K)."""

NA = 6.02214076e23
"""Avogadro constant (1/mol)."""

uSI = 1e-3 / NA
"""Atomic mass constant (kg)."""

# approximate constants in SI units:

meSI = 9.1093837015e-31
"""Electron mass (kg) [2018 CODATA]."""

eps0 = 8.8541878128e-12
"""Vacuum permittivity (F/m) [2018 CODATA]."""

# constants in atomic units (eV, AA):

Ry = 13.605693122994
"""Rydberg energy (eV) [2018 CODATA]."""

Ha = 2 * Ry
"""Hartree energy (eV)."""

a0 = 0.529177210903
"""Bohr radius (AA) [2018 CODATA]."""

cmm1 = 100.0 * hSI * cSI / eVSI
""""Inverse cm" (eV)."""

kB = kBSI / eVSI
"""Boltzmann constant (eV/K)."""

# constants in Rydberg atomic units:

uRy = uSI / (2 * meSI)
"""Atomic mass constant (2 me)."""

ohmRy = eVSI ** 2 / 2 / hbarSI
"""Resistance unit ohm (Rydberg atomic units)."""

ohmmRy = 1e10 / a0 * ohmRy
"""Resistivity unit ohm metre (Rydberg atomic units)."""

colors = collections.defaultdict(lambda: (250, 22, 145),
    H=(255, 255, 255), He=(217, 255, 255), Li=(204, 128, 255),
    Be=(194, 255, 0), B=(255, 181, 181), C=(144, 144, 144),
    N=(48, 80, 248), O=(255, 13, 13), F=(144, 224, 80),
    Ne=(179, 227, 245), Na=(171, 92, 242), Mg=(138, 255, 0),
    Al=(191, 166, 166), Si=(240, 200, 160), P=(255, 128, 0),
    S=(255, 255, 48), Cl=(31, 240, 31), Ar=(128, 209, 227),
    K=(143, 64, 212), Ca=(61, 255, 0), Sc=(230, 230, 230),
    Ti=(191, 194, 199), V=(166, 166, 171), Cr=(138, 153, 199),
    Mn=(156, 122, 199), Fe=(224, 102, 51), Co=(240, 144, 160),
    Ni=(80, 208, 80), Cu=(200, 128, 51), Zn=(125, 128, 176),
    Ga=(194, 143, 143), Ge=(102, 143, 143), As=(189, 128, 227),
    Se=(255, 161, 0), Br=(166, 41, 41), Kr=(92, 184, 209),
    Rb=(112, 46, 176), Sr=(0, 255, 0), Y=(148, 255, 255),
    Zr=(148, 224, 224), Nb=(115, 194, 201), Mo=(84, 181, 181),
    Tc=(59, 158, 158), Ru=(36, 143, 143), Rh=(10, 125, 140),
    Pd=(0, 105, 133), Ag=(192, 192, 192), Cd=(255, 217, 143),
    In=(166, 117, 115), Sn=(102, 128, 128), Sb=(158, 99, 181),
    Te=(212, 122, 0), I=(148, 0, 148), Xe=(66, 158, 176),
    Cs=(87, 23, 143), Ba=(0, 201, 0), La=(112, 212, 255),
    Ce=(255, 255, 199), Pr=(217, 255, 199), Nd=(199, 255, 199),
    Pm=(163, 255, 199), Sm=(143, 255, 199), Eu=(97, 255, 199),
    Gd=(69, 255, 199), Tb=(48, 255, 199), Dy=(31, 255, 199),
    Ho=(0, 255, 156), Er=(0, 230, 117), Tm=(0, 212, 82),
    Yb=(0, 191, 56), Lu=(0, 171, 36), Hf=(77, 194, 255),
    Ta=(77, 166, 255), W=(33, 148, 214), Re=(38, 125, 171),
    Os=(38, 102, 150), Ir=(23, 84, 135), Pt=(208, 208, 224),
    Au=(255, 209, 35), Hg=(184, 184, 208), Tl=(166, 84, 77),
    Pb=(87, 89, 97), Bi=(158, 79, 181), Po=(171, 92, 0),
    At=(117, 79, 69), Rn=(66, 130, 150), Fr=(66, 0, 102),
    Ra=(0, 125, 0), Ac=(112, 171, 250), Th=(0, 186, 255),
    Pa=(0, 161, 255), U=(0, 143, 255), Np=(0, 128, 255),
    Pu=(0, 107, 255), Am=(84, 92, 242), Cm=(120, 92, 227),
    Bk=(138, 79, 227), Cf=(161, 54, 212), Es=(179, 31, 212),
    Fm=(179, 31, 186), Md=(179, 13, 166), No=(189, 13, 135),
    Lr=(199, 0, 102), Rf=(204, 0, 89), Db=(209, 0, 79),
    Sg=(217, 0, 69), Bh=(224, 0, 56), Hs=(230, 0, 46),
    Mt=(235, 0, 38))
"""Jmol's element color scheme from http://jmol.sourceforge.net/jscolors/."""

class StatusBar:
    """Progress bar that does without carriage return or backspace.

    Parameters
    ----------
    count : int
        Number of times the progress bar will be updated.
    width : int, default 60
        Number of text columns the progress bar will span.
    title : str, default 'progress'
        Title line of the progress bar. Should be shorter than `width`.

    Attributes
    ----------
    in_progress : StatusBar
        Instance of the active progress bar or ``False`` for the first process;
        always ``True`` for the other processes. Used to ensure that only one
        progress bar is output at a time.
    """
    in_progress = comm.rank != 0

    def __init__(self, count, width=60, title='progress'):
        if not count or StatusBar.in_progress or verbosity < 3:
            return

        StatusBar.in_progress = self

        self.counter = 0
        self.count = count
        self.width = width
        self.progress = 0

        sys.stdout.write((' %s ' % title).center(width, '_'))
        sys.stdout.write('\n')

    def update(self):
        """Update progress bar."""

        if StatusBar.in_progress is not self:
            return

        self.counter += 1

        progress = self.width * self.counter // self.count

        if progress != self.progress:
            sys.stdout.write('=' * (progress - self.progress))
            sys.stdout.flush()

            self.progress = progress

        if self.counter == self.count:
            sys.stdout.write('\n')

            StatusBar.in_progress = False

def hello():
    if verbosity:
        info('This is elphmod (version %s) running on %d processors.'
            % (elphmod.__version__, comm.size))
        info('To suppress all output, set elphmod.misc.verbosity = 0.')

def get_sparse_array():
    """Try to import sparse array or matrix."""

    try:
        from scipy.sparse import dok_array as sparse_array
    except ImportError:
        try:
            from scipy.sparse import dok_matrix as sparse_array
        except ImportError:
            info('Sparse arrays require SciPy!', error=True)

    return sparse_array

def group(points, eps=1e-7):
    """Group points into neighborhoods.

    Parameters
    ----------
    points : ndarray
        Points to be grouped.
    eps : float
        Maximal distance between points in the same group.

    Returns
    -------
    list of lists
        Groups of indices.
    """
    groups = np.arange(len(points))

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if np.all(np.absolute(points[j] - points[i]) < eps):
                groups[np.where(groups == groups[j])] = groups[i]

    return [np.where(groups == group)[0] for group in set(groups)]

def differential(x):
    """Calculate weights for numerical integration with trapezoidal rule.

    Parameters
    ----------
    x : ndarray
        Sample points.

    Returns
    -------
    ndarray
        Integration Weights.
    """
    dx = np.empty_like(x)

    dx[0] = x[1] - x[0]
    dx[1:-1] = x[2:] - x[:-2]
    dx[-1] = x[-1] - x[-2]

    dx /= 2

    return dx

def rand(*shape, a=48271, m=2147483647):
    """Create array with MINSTD random values in a given shape.

    Parameters
    ----------
    *shape : int
        Array dimensions.
    a, m : int
        Parameters of Park–Miller random-number generator.

    Returns
    -------
    ndarray
        Random values.
    """
    array = np.empty(np.prod(shape))

    for n in range(array.size):
        rand.i *= a
        rand.i %= m

        array[n] = rand.i / m

    return array.reshape(shape)

rand.i = 1

def read_cube(cube, only_header=False, comm=comm):
    """Read Gaussian cube file.

    Parameters
    ----------
    cube : str
        Name of Gaussian cube file.
    only_header : bool, default False
        Skip reading data?

    Returns
    -------
    r0 : ndarray
        Origin of the data grid.
    a : ndarray
        Spanning vectors of the data grid.
    X : list of str
        Atomic numbers.
    tau : ndarray
        Cartesian atomic coordinates.
    data : ndarray
        Data-grid values or, if `only_header`, shape of data grid.

    See Also
    --------
    read_xsf : Equivalent function for XCrySDen format.
    """
    if comm.rank == 0:
        lines = open(cube)

        next(lines)
        next(lines)

        cols = next(lines).split()
        nat = int(cols[0])
        r0 = np.array(list(map(float, cols[1:4])))
    else:
        nat = None

    nat = comm.bcast(nat)

    n = np.empty(3, dtype=int)
    r0 = np.empty(3, dtype=float)
    a = np.empty((3, 3), dtype=float)
    X = np.empty(nat, dtype=int)
    tau = np.empty((nat, 3), dtype=float)

    if comm.rank == 0:
        for i in range(3):
            cols = next(lines).split()
            n[i] = int(cols[0])
            a[i] = list(map(float, cols[1:4]))
            a[i] *= n[i]

        for i in range(nat):
            cols = next(lines).split()
            X[i] = int(cols[0])
            tau[i] = list(map(float, cols[2:5]))

    comm.Bcast(n)
    comm.Bcast(r0)
    comm.Bcast(a)
    comm.Bcast(X)
    comm.Bcast(tau)

    if only_header:
        return r0, a, X, tau, tuple(n)

    if comm.rank == 0:
        data = np.empty(np.prod(n))

        i = 0
        for line in lines:
            for number in line.split():
                data[i] = float(number)
                i += 1

        data = np.reshape(data, n)

        lines.close()
    else:
        data = np.empty(n, dtype=float)

    comm.Bcast(data)

    return r0, a, X, tau, data

def read_xsf(xsf, only_header=False, comm=comm):
    """Read file in XCrySDen format.

    Parameters
    ----------
    xsf : str
        Name of XCrySDen file.
    only_header : bool, default False
        Skip reading data?

    Returns
    -------
    r0 : ndarray
        Origin of the data grid.
    a : ndarray
        Spanning vectors of the data grid.
    X : list of str
        Atomic numbers or symbols.
    tau : ndarray
        Cartesian atomic coordinates.
    data : ndarray
        Data-grid values or, if `only_header`, shape of data grid.

    See Also
    --------
    read_cube : Equivalent function for Gaussian cube format.
    write_xsf : Write file in XCrySDen format.
    """
    if comm.rank == 0:
        lines = open(xsf)

        while next(lines).strip() != 'PRIMCOORD':
            pass

        nat = int(next(lines).split()[0])
    else:
        nat = None

    nat = comm.bcast(nat)

    n = np.empty(3, dtype=int)
    r0 = np.empty(3, dtype=float)
    a = np.empty((3, 3), dtype=float)
    X = ['X'] * nat
    tau = np.empty((nat, 3), dtype=float)

    if comm.rank == 0:
        for i in range(nat):
            cols = next(lines).split()
            X[i] = cols[0]
            tau[i] = list(map(float, cols[1:4]))

        while next(lines).strip() != 'BEGIN_DATAGRID_3D_UNKNOWN':
            pass

        n[:] = list(map(int, next(lines).split()))
        r0[:] = list(map(float, next(lines).split()))

        for i in range(3):
            a[i] = list(map(float, next(lines).split()))

    comm.Bcast(n)
    comm.Bcast(r0)
    comm.Bcast(a)
    X = comm.bcast(X)
    comm.Bcast(tau)

    if only_header:
        return r0, a, X, tau, tuple(n)

    if comm.rank == 0:
        data = np.empty(np.prod(n))

        i = 0
        while i < data.size:
            for number in next(lines).split():
                data[i] = float(number)
                i += 1

        data = np.reshape(data, n[::-1])
        data = np.transpose(data).copy()

        lines.close()
    else:
        data = np.empty(n, dtype=float)

    comm.Bcast(data)

    return r0, a, X, tau, data

def write_xsf(xsf, r0, a, X, tau, data, only_header=False, comm=comm):
    """Write file in XCrySDen format.

    Parameters
    ----------
    xsf : str
        Name of XCrySDen file.
    r0 : ndarray
        Origin of the data grid.
    a : ndarray
        Spanning vectors of the data grid.
    X : list of str
        Atomic numbers or symbols.
    tau : ndarray
        Cartesian atomic coordinates.
    data : ndarray
        Data-grid values or, if `only_header`, shape of data grid.
    only_header : bool, default False
        Skip writing data?

    See Also
    --------
    read_xsf : Read file in XCrySDen format.
    """
    if comm.rank == 0:
        n = tuple(data) if only_header else data.shape

        with open(xsf, 'w') as text:
            text.write('PRIMCOORD\n')
            text.write('%6d  1\n' % len(X))

            for i in range(len(X)):
                text.write('%-5s' % X[i])
                text.write(' %11.7f %11.7f %11.7f\n' % tuple(tau[i]))

            text.write('BEGIN_BLOCK_DATAGRID_3D\n')
            text.write('3D_field\n')
            text.write('BEGIN_DATAGRID_3D_UNKNOWN\n')
            text.write('%6d %5d %5d\n' % tuple(n))
            text.write('%12.6f %11.6f %11.6f\n' % tuple(r0))

            for i in range(3):
                text.write('%12.7f %11.7f %11.7f\n' % tuple(a[i]))

            if not only_header:
                for i, value in enumerate(data.flat, 1):
                    text.write('%13.5e' % value)

                    if not i % 6 or i == data.size:
                        text.write('\n')

            text.write('END_DATAGRID_3D\n')
            text.write('END_BLOCK_DATAGRID_3D\n')

def real_space_grid(shape, r0, a, shared_memory=False):
    """Sample real-space grid.

    Parameters
    ----------
    shape : tuple of int
        Shape of the 3D real-space grid.
    r0 : ndarray
        Origin of the real-space grid.
    a : ndarray
        Cartesian spanning vectors of the real-space grid.
    shared_memory : bool, default False
        Store real-space grid in shared memory?

    Returns
    -------
    ndarray
        Cartesian coordinates for all grid points.
    """
    node, images, r = elphmod.MPI.shared_array((*shape, 3),
        shared_memory=shared_memory)

    if comm.rank == 0:
        axes = [np.linspace(0.0, 1.0, num) for num in shape]
        axes = np.meshgrid(*axes, indexing='ij')
        r[...] = r0 + np.einsum('nijk,nx->ijkx', axes, a)

    if node.rank == 0:
        images.Bcast(r)

    comm.Barrier()

    return r

def read_namelists(filename):
    """Extract all Fortran namelists from file.

    Parameters
    ----------
    filename : str
        Name of file with namelists.

    Returns
    -------
    dict of dict
        Namelist data.
    """
    import re

    with open(filename) as data:
        text = data.read()

    # remove comments:

    text = re.sub(r'!.*$', '', text, flags=re.MULTILINE)

    # protect string (which might contain spaces):

    groups = []

    def replace(match):
        groups.append(match.group(0))
        return '___%d___' % len(groups)

    def place(match):
        return groups[int(match.group(0).strip('_')) - 1]

    count = 1
    while count:
        text, count = re.subn(r'\'[^\'"]*\'|"[^\'"]*"', replace, text)

    # parse all namelists:

    data = dict()

    for namelist in re.finditer(r'&(\w+)(.+?)/', text, flags=re.DOTALL):
        name, content = namelist.groups()
        name = name.lower()
        data[name] = dict()

        content = re.sub(r'=', ' = ', content)
        content = re.sub(r'\s+\(', '(', content)
        items = re.split(r'[;,\s]+', content)

        key = None
        for i, item in enumerate(items):
            if not item:
                continue

            if '=' in items[i + 1:i + 2]:
                key, n = re.match(r'([^(]*)[( ]*(\d*)', item.lower()).groups()
                n = int(n) - 1 if n else 0

                if key not in data[name]:
                    data[name][key] = []

                while len(data[name][key]) < n:
                    data[name][key].append(None)

            elif key is not None and item != '=':
                count = 1
                while count:
                    item, count = re.subn(r'___\d+___', place, item)

                if re.match(r'(\'.*\'|".*")$', item):
                    item = item[1:-1]
                elif re.search('[Tt]', item):
                    item = True
                elif re.search('[Ff]', item):
                    item = False
                else:
                    try:
                        item = int(item)
                    except ValueError:
                        try:
                            item = float(re.sub(r'[dDE]', 'e', item))
                        except ValueError:
                            pass

                data[name][key][n:n + 1] = [item]
                n += 1

        for key, value in data[name].items():
            if len(value) == 1:
                data[name][key] = value[0]

    return data

def read_input_data(filename, broadcast=True):
    """Read Quantum ESPRESSO input data.

    Parameters
    ----------
    filename : str
        Name of input file.
    broadcast : bool
        Broadcast result from rank 0 to all processes?

    Returns
    -------
    dict
        Input data.
    """
    if comm.rank == 0:
        struct = dict()

        for namelist in read_namelists(filename).values():
            struct.update(namelist)
    else:
        struct = None

    if broadcast:
        struct = comm.bcast(struct)

    return struct

def read_dat_mat(filename):
    r"""Read matrix elements from RESPACK (*dat.Wmat*, *dat.h_mat_r*).

    Parameters
    ----------
    filename : str
        Name of the file.

    Returns
    -------
    ndarray
        Lattice vectors.
    ndarray
        Direct (screened) Coulomb or hoppling matrix elements, depending on the
        input file.
    """
    with open(filename) as respack_file:
        lines = respack_file.readlines()

    for line in range(3, len(lines)):
        if not lines[line].strip():
            num_wann = int(lines[line - 1].split()[0])
            break

    block = 1 + num_wann ** 2 + 1
    nR = int((len(lines) - 3) / block) # number of lattice vectors R
    R = np.empty((nR, 3), dtype=int)
    Rcount = 0

    # allocate W or t matrix:

    data = np.empty((nR, num_wann, num_wann), dtype=complex)

    for line in range(3, len(lines)):
        # read lattice vectors R:

        if len(lines[line].split()) == 3:
            R1, R2, R3 = lines[line].split()

            R[Rcount, 0] = int(R1)
            R[Rcount, 1] = int(R2)
            R[Rcount, 2] = int(R3)

        # read matrix elements:

        if len(lines[line].split()) == 4:
            n, m, real, imag = lines[line].split()

            n = int(n) - 1
            m = int(m) - 1

            real = float(real)
            imag = float(imag)

            data[Rcount, n, m] = real + 1j * imag

        if len(lines[line].split()) == 0:
            Rcount += 1

    return R, data

def split(expr, sd=',', od='{', cd='}'):
    """Split expression with separators and brackets using distributive law.

    Parameters
    ----------
    expr : str
        Expression to be expanded and split.
    sd : str
        Separating delimiter.
    od : str
        Opening delimiter.
    cd : str
        Closing delimiter.

    Returns
    -------
    generator
        Separated elements of expression.
    """
    import re

    group = '[{0}]([^{0}{1}]*)[{1}]'.format(*
        [re.sub(r'([\^\]])', r'\\\1', d) for d in (od, cd)])

    groups = []

    def pack(match):
        groups.append(match.group(1))
        return '<%d>' % (len(groups) - 1)

    expr = od + expr + cd

    n = 1
    while n:
        expr, n = re.subn(group, pack, expr)

    def factorize(x):
        match = re.match(r'(.*)<(\d+)>(.*)', x)

        if match:
            for y in groups[int(match.group(2))].split(sd):
                for z in factorize(match.group(1) + y.strip() + match.group(3)):
                    yield z
        else:
            yield x

    return factorize(expr)

def vector_index(vectors, vector):
    """Find index of vector in list of vectors.

    Parameters
    ----------
    vectors : ndarray
        List of vectors.
    vector : ndarray
        Vector.

    Returns
    -------
    int
        Index of vector in list of vectors.
    """
    match = np.all(vectors == vector, axis=1)

    if np.any(match):
        return np.argmax(match)
    else:
        return None

def minimum(*args):
    """Find extremum of parabola through given points.

    Parameters
    ----------
    *args
        Coordinates of three points. Can also be provided as command arguments.

    Returns
    -------
    float
        x coordinates of extremum.
    """
    if len(args) == 6:
        x1, x2, x3, y1, y2, y3 = args
    if len(sys.argv) == 7:
        x1, x2, x3, y1, y2, y3 = list(map(float, sys.argv[1:]))
    else:
        raise SystemExit('Usage: minimum x1 x2 x3 y1 y2 y3')

    enum = x1 ** 2 * (y2 - y3) + x2 ** 2 * (y3 - y1) + x3 ** 2 * (y1 - y2)
    deno = x1 * 2 * (y2 - y3) + x2 * 2 * (y3 - y1) + x3 * 2 * (y1 - y2)

    if deno:
        x0 = enum / deno
        info(x0)
        return x0
    else:
        info('There is no extremum.')
