# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

"""Charge-density-wave dynamics on supercells."""

import copy
import numpy as np
import time

import elphmod.bravais
import elphmod.diagrams
import elphmod.dispersion
import elphmod.el
import elphmod.misc
import elphmod.MPI
import elphmod.occupations
import elphmod.ph

comm = elphmod.MPI.comm
info = elphmod.MPI.info

class Driver:
    """MD driver for DFPT-based displacements dynamics.

    Parameters
    ----------
    elph : :class:`elphmod.elph.Model`
        Localized model for electron-phonon coupling. Initialize ``el`` with
        ``rydberg=True`` and ``ph`` and ``elph`` with ``divide_mass=False`` and
        map everthing to the appropriate supercell before (``elph.supercell``).
    kT : float
        Smearing temperature in Ry.
    f : function
        Particle distribution as a function of energy divided by `kT`.
    n : float
        Number of electrons per primitive cell.
    nx : float, default 0.0
        Number of photoexcited electrons per primitive cell.
    nk, nq : tuple of int, optional
        Shape of k and q mesh. By default, only k = q = 0 is used.
    supercell : ndarray, optional
        Supercell lattice vectors as multiples of primitive lattice vectors. If
        given, the simulation is performed on a supercell for q = k = 0. Sparse
        matrices are used for Hamiltonian, dynamical matrix, and electron-phonon
        coupling to save memory. Note that `elph` should belong to the primitive
        cell in this case, and `nq` and `nk` are only used for the unscreening,
        which is still done one the primitive cell. To do the unscreening on the
        supercell (recommended), a sparse supercell model `elph` can be directly
        provided instead of using this option.
    unscreen : bool, default True
        Unscreen phonons? Otherwise, they are assumed to be unscreened already.
    shared_memory : bool, default True
        Store :attr:`d0` and :attr:`d` in shared memory?
    **kwargs
        Attributes to be set initially.

    Attributes
    ----------
    elph, kT, f, n, nk, nq
        Copies of initialization parameters.
    mu : float
        Current chemical potential.
    k, q : ndarray
        k and q meshes.
    u : ndarray
        Atomic displacements.
    C0 : ndarray
        Unscreened force constants.
    H0 : ndarray
        Unperturbed electron Hamiltonian in orbital basis.
    d0 : ndarray
        Electron-phonon coupling in orbital basis.
    d : ndarray
        Electron-phonon coupling in band basis.
    sparse : bool
        Is the simulation performed on a supercell using sparse matrices?
    interactive : bool, default False
        Shall plots be updated interactively?
    scale : float, default 10.0
        Displacement scaling factor for plots.
    size : float, default 100.0
        Marker size for atoms in points squared.
    pause : float, default 1e-3
        Minimal frame duration for interactive plots in seconds.
    basis : list of list, default None
        For each basis atom in the first primitive cell, indices of orbitals
        located at this atom. Matching atom and orbital orders as ensured by
        :meth:`elphmod.elph.Model.supercell` are required.
    node, images : MPI.Intracomm
        Communicators between processes that share memory or same ``node.rank``
        if `shared_memory`.
    """
    def __init__(self, elph, kT, f, n, nx=0.0, nk=(1,), nq=(1,),
            supercell=None, unscreen=True, shared_memory=True, **kwargs):

        if not elph.el.rydberg:
            info("Initialize 'el' with 'rydberg=True'!", error=True)

        if elph.divide_mass or elph.ph.divide_mass:
            info("Initialize 'ph' and 'elph' with 'divide_mass=False'!",
                error=True)

        self.elph = copy.copy(elph)

        self.kT = kT
        self.f = elphmod.occupations.smearing(f)

        self.n = n
        self.nx = nx
        self.mu = None

        self.nk = np.ones(3, dtype=int)
        self.nk[:len(nk)] = nk

        self.nq = np.ones(3, dtype=int)
        self.nq[:len(nq)] = nq

        self.k = elphmod.bravais.mesh(*self.nk)
        self.q = elphmod.bravais.mesh(*self.nq, flat=True)

        try:
            self.H0 = self.elph.el.Hs.toarray()
            self.C0 = self.elph.ph.Ds.toarray()[np.newaxis]
            self.d0 = self.elph.gs

            if self.nk.prod() != 1 or self.nq.prod() != 1:
                info('MD using sparse matrices requires q = k = 0!', error=True)

            self.sparse = True

        except AttributeError:
            self.H0 = elphmod.dispersion.sample(self.elph.el.H, self.k)
            self.C0 = elphmod.dispersion.sample(self.elph.ph.D, self.q)

            self.d0 = self.elph.sample(self.q, self.nk,
                shared_memory=shared_memory)

            self.node, self.images, self.d = elphmod.MPI.shared_array(
                self.d0.shape, dtype=self.d0.dtype, shared_memory=shared_memory)

            self.sparse = False

        self.u = np.zeros(self.elph.ph.size)

        self.diagonalize()

        if unscreen:
            self.C0 -= self.hessian(gamma_only=False) - self.C0

        self.F0 = np.zeros(self.elph.ph.size)
        self.F0 = -self.jacobian(show=False)

        self.interactive = False
        self.scale = 10.0
        self.size = 100.0
        self.pause = 1e-3
        self.basis = None

        for name, value in kwargs.items():
            setattr(self, name, value)

        if supercell is not None:
            if unscreen:
                self.elph.ph = copy.copy(self.elph.ph)

                elphmod.ph.q2r(self.elph.ph, nq=self.nq, D_full=self.C0,
                    divide_mass=False)

            elph = self.elph.supercell(*supercell, sparse=True)

            self.__init__(elph, self.kT, self.f, self.n * len(elph.cells),
                self.nx * len(elph.cells), unscreen=False, **kwargs)

    def random_displacements(self, amplitude=0.01, reproducible=False):
        """Displace atoms randomly from unperturbed positions.

        Parameters
        ----------
        amplitude : float, default 0.01
            Maximum displacement.
        reproducible : bool, default False
            Use same random numbers in each program run?
        """
        if comm.rank == 0:
            rand = elphmod.misc.rand if reproducible else np.random.rand
            self.u = amplitude * (1 - 2 * rand(self.u.size))
            self.center_mass()

        comm.Bcast(self.u)

    def center_mass(self):
        """Subtract collective translational displacement component."""

        self.u -= np.tile(np.average(self.u.reshape((-1, 3)), axis=0),
            self.elph.ph.nat)

    def find_Fermi_level(self):
        """Update chemical potential.

        Returns
        -------
        float
            New chemical potential.
        """
        if self.f is elphmod.occupations.two_fermi_dirac:
            if self.n % 2:
                info('Photoexcitation requires even number of electrons!',
                    error=True)

            muv = elphmod.occupations.find_Fermi_level(self.n - self.nx,
                self.e[..., :int(self.n) // 2], self.kT,
                elphmod.occupations.fermi_dirac, self.mu)

            muc = elphmod.occupations.find_Fermi_level(self.nx,
                self.e[..., int(self.n) // 2:], self.kT,
                elphmod.occupations.fermi_dirac, self.mu)

            self.f.d = (muc - muv) / (2 * self.kT)
            self.mu = (muc + muv) / 2
        else:
            self.mu = elphmod.occupations.find_Fermi_level(self.n,
                self.e, self.kT, self.f, self.mu)

        return self.mu

    def diagonalize(self):
        """Diagonalize Hamiltonian of perturbed system."""

        self.center_mass()

        if self.sparse:
            H = self.H0 + self.u.dot(self.d0).toarray()
        else:
            H = self.H0 + np.einsum('xijkmn,x->ijkmn', self.d0[0], self.u)

        self.e, self.U = np.linalg.eigh(H)

        self.e -= self.find_Fermi_level()

    def free_energy(self, u=None, show=True):
        """Calculate free energy.

        Parameters
        ----------
        u : ndarray
            Updated atomic displacements (e.g., from optimization routine).
        show : bool
            Print free energy?

        Returns
        -------
        float
            Free energy in Ry.
        """
        if show:
            t0 = time.time()

        if u is not None:
            self.u = u

        self.diagonalize()

        if self.interactive:
            self.update_plot()

        E = elphmod.diagrams.grand_potential(self.e,
            self.kT, self.f) + self.mu * self.n

        E += 0.5 * self.u.dot(self.C0[0].real).dot(self.u)

        E += self.F0.dot(self.u)

        if show:
            info('Free energy: %15.9f Ry; %16.6f s' % (E, time.time() - t0))

        return E

    def jacobian(self, parameters=None, show=True):
        """Calculate first derivative of free energy.

        Parameters
        ----------
        parameters : ndarray
            Dummy positional argument for optimization routines.
        show : bool
            Print free energy?

        Returns
        -------
        ndarray
            Negative forces in Ry per bohr.
        """
        if show:
            t0 = time.time()

        if self.sparse:
            f = self.U * self.f(self.e / self.kT)[np.newaxis] @ self.U.T

            F = np.array([2 * self.d0[x].multiply(f).sum()
                for x in range(self.elph.ph.size)])
        else:
            F = elphmod.diagrams.first_order(self.e, self.d0[0], self.kT,
                U=self.U, occupations=self.f).real

        F += self.C0[0].real.dot(self.u)

        F += self.F0

        if show:
            info('Total force: %15.9f Ry/bohr; %11.6f s'
                % (np.linalg.norm(F), time.time() - t0))

        return F

    def hessian(self, parameters=None, gamma_only=True, apply_asr_simple=False,
            fildyn=None, eps=1e-10, kT=None):
        """Calculate second derivative of free energy.

        Parameters
        ----------
        parameters : ndarray
            Dummy positional argument for optimization routines.
        gamma_only : bool, default True
            Calculate Hessian for q = 0 only?
        apply_asr_simple : bool, default False
            Apply simple acoustic sum rule correction to force constants
            according to Eq. 81 of Gonze and Lee, Phys. Rev. B 55, 10355 (1997)?
            This is done before saving the Hessian to file.
        fildyn : str, optional
            Filename to save Hessian.
        eps : float
            Smallest allowed absolute value of divisor.
        kT : float, optional
            Smearing temperature. If given, it is used to calculate the double
            Fermi-surface average of the electron-phonon coupling squared, which
            is then also returned.

        Returns
        -------
        ndarray
            Force constants in Ry per bohr squared.
        ndarray, optional
            Fermi-surface average of electron-phonon coupling squared.
        """
        nq = 1 if gamma_only or self.sparse else len(self.q)

        if self.sparse:
            x = self.e / self.kT
            f = 2 * self.f(x)
            d = 2 * self.f.delta(x) / (-self.kT)

            df = np.subtract.outer(f, f)
            de = np.subtract.outer(self.e, self.e)
            ok = abs(de) > eps

            dfde = np.tile(d, (self.elph.el.size, 1))
            dfde[ok] = df[ok] / de[ok]
            dos = -d.sum() # = -np.trace(dfde)

            if kT is not None:
                tmp = d if kT == self.kT else self.f.delta(self.e / kT) / kT
                dd = np.outer(tmp, tmp)
                dd /= dd.sum()

                g2dd = np.empty((nq, self.elph.ph.size, self.elph.ph.size))

            C = np.zeros((nq, self.elph.ph.size, self.elph.ph.size))

            V = self.U.transpose().copy()

            d_orb = self.U * d[np.newaxis] @ V
            avg = np.array([self.d0[x].multiply(d_orb).sum()
                for x in range(self.elph.ph.size)]) / np.sqrt(dos)

            status = elphmod.misc.StatusBar(self.elph.ph.size,
                title='calculate static phonon self-energy')

            for x in range(self.elph.ph.size):
                gx = V.dot(self.d0[x].dot(self.U))

                if kT is not None:
                    gxdd = self.U @ (gx * dd) @ V

                gx = self.U @ (gx * dfde) @ V # overwriting gx to save memory

                for y in range(x, self.elph.ph.size):
                    C[0, x, y] = self.d0[y].multiply(gx).sum() + avg[x] * avg[y]
                    C[0, y, x] = C[0, x, y]

                    if kT is not None:
                        g2dd[0, x, y] = self.d0[y].multiply(gxdd).sum()
                        g2dd[0, y, x] = g2dd[0, x, y]

                status.update()
        else:
            for iq in range(nq):
                if comm.rank == 0:
                    V = self.U.conj().swapaxes(-2, -1)

                    q = np.round(self.nk * self.q[iq] / (2 * np.pi)).astype(int)

                    for i in range(3):
                        if q[i]:
                            V = np.roll(V, -q[i], axis=i)

                    self.d[iq] = V @ self.d0[iq] @ self.U

                if self.node.rank == 0:
                    self.images.Bcast(self.d[iq].view(dtype=float))

            C = elphmod.diagrams.phonon_self_energy(self.q[:nq], self.e,
                g=self.d[:nq], kT=self.kT, occupations=self.f, eps=eps)

            C[0] += elphmod.diagrams.phonon_self_energy_fermi_shift(self.e,
                self.d[0], kT=self.kT, occupations=self.f)

        C += self.C0[:nq]

        if apply_asr_simple:
            C = C.reshape((nq, self.elph.ph.nat, 3, self.elph.ph.nat, 3))

            corr = C.sum(axis=3)

            for na in range(self.elph.ph.nat):
                C[:, na, :, na, :] -= corr[:, na]

            C = C.reshape((nq, self.elph.ph.size, self.elph.ph.size))

        if fildyn is not None and comm.rank == 0:
            b = elphmod.bravais.reciprocals(*self.elph.ph.a)

            elphmod.ph.write_flfrc(fildyn, (self.q[:nq].dot(b), C),
                self.elph.ph.M, self.elph.ph.a,
                self.elph.ph.r + self.u.reshape(self.elph.ph.r.shape),
                self.elph.ph.atom_order)

        returns = C[0].real if gamma_only else C

        if kT is not None:
            returns = returns, g2dd

        return returns

    def electrons(self, seedname=None, dk1=1, dk2=1, dk3=1, rydberg=False):
        """Set up tight-binding model for current structure.

        Parameters
        ----------
        seedname : str
            Prefix of file with Hamiltonian in Wannier90 format.
        dk1, dk2, dk3 : int, optional
            Only use data for every `dkn`-th k point along the *n*-th axis? This
            reduces the size of the Hamiltonian file.
        rydberg : bool, default False
            Keep Ry units? Otherwise they are converted to eV.

        Returns
        -------
        object
            Tight-binding model for the electrons.
        """
        H = self.U * self.e[..., np.newaxis, :] @ self.U.conj().swapaxes(-2, -1)

        if dk1 > 1 or dk2 > 1 or dk3 > 1:
            if not self.sparse:
                H = H[::dk1, ::dk2, ::dk3]

        el = copy.deepcopy(self.elph.el)
        el.rydberg = rydberg

        if hasattr(self.elph.el, 'cells'):
            assert self.elph.el.cells == self.elph.ph.cells

            r = self.elph.ph.r[::self.elph.ph.nat // len(self.elph.ph.cells)]
            r = np.repeat(r, self.elph.el.size // r.shape[0], axis=0)
            r -= r[0]
        else:
            r = np.zeros((self.elph.el.size, 3))

        elphmod.el.k2r(el, H, self.elph.ph.a, r, rydberg=True)

        el.standardize(eps=1e-10)

        if seedname is not None:
            el.to_hrdat(seedname)

        return el

    def phonons(self, divide_mass=True, **kwargs):
        """Set up mass-spring model for current structure.

        Parameters
        ----------
        divide_mass : bool
            Divide force constants by atomic masses?
        **kwargs
            Parameters passed to :func:`elphmod.ph.q2r`.

        Returns
        -------
        object
            Mass-spring model for the phonons.
        """
        ph = copy.deepcopy(self.elph.ph)
        ph.divide_mass = divide_mass
        ph.r += self.u.reshape((-1, 3))

        elphmod.ph.q2r(ph, D_full=self.hessian(gamma_only=False), nq=self.nq,
            divide_mass=False, **kwargs)

        return ph

    def superconductivity(self, eps=1e-10, kT=None):
        r"""Calculate effective couplings and phonon frequencies.

        Note that :attr:`d` is destroyed.

        Parameters
        ----------
        eps : float
            Phonon frequencies squared below `eps` are set to `eps`;
            corresponding couplings are set to zero.
        tol : float, optional
            If any phonon frequency squared is smaller than `tol`, all return
            values are ``None``. A very small negative value can be chosen to
            skip calculations involving significant imaginary frequencies.
        kT : float, optional
            Smearing temperature. By default, :attr:`kT` is used.

        Returns
        -------
        float
            Effective electron-phonon coupling strength :math:`\lambda`.
        float
            Logarithmic average phonon energy.
        float
            Second-moment average phonon energy.
        float
            Minimum phonon energy. Imaginary frequencies are given as negative.
        """
        if kT is None:
            kT = self.kT

        mm12 = 1 / np.sqrt(self.elph.ph.M).repeat(3)

        if self.sparse:
            D, g2dd = self.hessian(gamma_only=False, kT=kT)
        else:
            D = self.hessian(gamma_only=False)

        D *= mm12[np.newaxis, np.newaxis, :]
        D *= mm12[np.newaxis, :, np.newaxis]

        w2, u = np.linalg.eigh(D)

        wmin = elphmod.ph.sgnsqrt(w2.min())

        if self.sparse:
            g2dd *= mm12[np.newaxis, np.newaxis, :]
            g2dd *= mm12[np.newaxis, :, np.newaxis]
            g2dd = np.diagonal(u.swapaxes(-2, -1) @ g2dd @ u,
                axis1=1, axis2=2).copy()
        else:
            for iq in range(len(self.q)):
                if self.node.rank == iq % self.node.size:
                    self.d[iq] *= mm12[:,
                        np.newaxis, np.newaxis, np.newaxis,
                        np.newaxis, np.newaxis]

                    self.d[iq] = np.sum(self.d[iq, :, np.newaxis] * u[iq, :, :,
                        np.newaxis, np.newaxis, np.newaxis,
                        np.newaxis, np.newaxis], axis=0)

                    self.d[iq] *= self.d[iq].conj()

            comm.Barrier()

            g2dd, dd = elphmod.diagrams.double_fermi_surface_average(self.q,
                self.e, self.d, kT, self.f)

            g2dd = g2dd.real / dd.sum()

        dangerous = np.where(w2 < eps)
        w2[dangerous] = eps
        g2dd[dangerous] = 0.0

        N0 = self.f.delta(self.e / kT).sum() / kT / self.nk.prod()

        V = g2dd / w2

        lamda = N0 * V.sum()
        wlog = np.exp((V * np.log(w2) / 2).sum() / V.sum())
        w2nd = np.sqrt((V * w2).sum() / V.sum())

        return lamda, wlog, w2nd, wmin

    def density(self):
        """Calculate electron density for all orbitals.

        Returns
        -------
        ndarray
            Electron density. Should add up to the number of electrons.
        """
        return 2 * np.sum(np.average(np.reshape(abs(self.U) ** 2
            * self.f(self.e[..., np.newaxis, :] / self.kT),
            (-1, self.elph.el.size, self.elph.el.size)), axis=0), axis=-1)

    def density_per_atom(self):
        """Calculate electron density per atom."""

        if self.basis is None:
            info('Orbitals located at basis atoms unknown!', error=True)

        nat = len(self.basis)
        norb = self.elph.el.size * nat // self.elph.ph.nat

        rho_at = np.zeros(self.elph.ph.nat)
        rho_orb = self.density()

        for na, orbitals in enumerate(self.basis):
            for no in orbitals:
                rho_at[na::nat] += rho_orb[no::norb]

        return rho_at

    def plot(self, filename=None, interactive=None, scale=None, padding=1.0,
            size=None, pause=None, label=False, elev=None, azim=None):
        """Plot crystal structure and displacements.

        Parameters
        ----------
        filename : str, optional
            Figure filename. If given, the plot is saved rather than shown.
        interactive : bool, optional
            Shall the plot be updated? If given, this sets the eponymous
            attribute, which is used by default.
        scale : float, optional
            Displacement scaling factor. If given, this sets the eponymous
            attribute, which is used by default.
        padding : float, optional
            Padding between crystal and plotting box in angstrom.
        size : float, optional
            Marker size for atoms.
        pause : float, optional
            Minimal frame duration for interactive plots in seconds.
        label : bool, optional
            Show atom indices?
        elev, azim : float, optional
            Elevation and azimuthal view angles.
        """
        if comm.rank != 0:
            return

        global plt
        import matplotlib.pyplot as plt

        if interactive is not None:
            self.interactive = interactive

        if scale is not None:
            self.scale = scale

        if size is not None:
            self.size = size

        if pause is not None:
            self.pause = pause

        u = self.u.reshape(self.elph.ph.r.shape).T
        r = self.elph.ph.r.T + u

        if elev is None:
            elev = self.axes.elev if hasattr(self, 'axes') else 90

        if azim is None:
            azim = self.axes.azim if hasattr(self, 'axes') else -90

        self.axes = plt.axes(projection='3d')
        self.axes.view_init(elev, azim)

        sizes = self.size

        if self.basis is not None:
            rho = self.density_per_atom()
            sizes *= rho / rho.max()

        self.scatter = self.axes.scatter(*r, s=sizes, c=['#%02x%02x%02x'
            % elphmod.misc.colors[X] for X in self.elph.ph.atom_order])

        self.quiver = self.axes.quiver(*r, *self.scale * u, color='gray')

        if label:
            for na in range(self.elph.ph.nat):
                self.axes.text(*r[:, na], str(na))

        lims = [self.axes.set_xlim, self.axes.set_ylim, self.axes.set_zlim]

        for i, lim in enumerate(lims):
            lim(self.elph.ph.r[:, i].min() - padding,
                self.elph.ph.r[:, i].max() + padding)

        self.axes.set_box_aspect(np.ptp(self.elph.ph.r, axis=0) + 2 * padding)
        self.axes.set_axis_off()

        if self.interactive:
            plt.ion()
        else:
            plt.ioff()

        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

    def update_plot(self):
        """Update open plot."""

        if comm.rank != 0:
            return

        u = self.u.reshape(self.elph.ph.r.shape).T
        r = self.elph.ph.r.T + u

        self.scatter._offsets3d = tuple(r)

        if self.basis is not None:
            rho = self.density_per_atom()
            self.scatter.set_sizes(self.size * rho / rho.max())

        self.quiver.remove()
        self.quiver = self.axes.quiver(*r, *self.scale * u, color='gray')

        plt.pause(self.pause)

    def to_xyz(self, xyz, append=False):
        """Save current atomic positions.

        Parameters
        ----------
        xyz : str
            Name of .xyz file.
        append : bool, default False
            Append rather than overwrite positions?
        """
        if comm.rank == 0:
            with open(xyz, 'a' if append else 'w') as data:
                data.write('%d\n' % self.elph.ph.nat)
                data.write(('# CELL{H}:' + ' %.10g' * 9 + '\n')
                    % tuple(self.elph.ph.a.ravel(order='F')))

                pos = self.elph.ph.r + self.u.reshape(self.elph.ph.r.shape)

                for X, r in zip(self.elph.ph.atom_order, pos):
                    data.write(('%8s' + ' %15.9f' * 3 + '\n')
                        % (X, r[0], r[1], r[2]))

    def from_xyz(self, xyz):
        """Load saved atomic positions if compatible.

        If an interactive plot is open, it is updated. If `xyz` contains a full
        trajectory, all steps are shown in an interactive plot (whose speed can
        be controlled via :attr:`pause`) and the last atomic positions are kept.

        Parameters
        ----------
        xyz : str
            Name of .xyz file.
        """
        if comm.rank == 0:
            with open(xyz) as lines:
                for line in lines:
                    nat = int(line)

                    if nat != self.elph.ph.nat:
                        print("Error: Wrong number of atoms in '%s'!" % xyz)
                        break

                    next(lines) # skip comment line

                    for na in range(nat):
                        cols = next(lines).split()

                        if cols[0] != self.elph.ph.atom_order[na]:
                            print("Warning: Unexpected atom %d in '%s' ignored!"
                                % (na + 1, xyz))
                            continue

                        pos = np.array(list(map(float, cols[1:4])))

                        self.u[elphmod.ph.group(na)] = pos - self.elph.ph.r[na]

                    if self.interactive:
                        self.update_plot()

        comm.Bcast(self.u)

    def to_pwi(self, pwi, **kwargs):
        """Save current atomic positions etc. to PWscf input file.

        Parameters
        ----------
        pwi : str
            Filename.
        **kwargs
            Keyword arguments with further parameters to be written.
        """
        pw = dict()

        pw['r'] = elphmod.bravais.cartesian_to_crystal(self.elph.ph.r
            + self.u.reshape(self.elph.ph.r.shape), *self.elph.ph.a)

        if self.nk.prod() == 1:
            pw['ktyp'] = 'gamma'
        else:
            pw['ktyp'] = 'automatic'
            pw['k_points'] = (*self.nk, 0, 0, 0)

        pw.update(kwargs)

        self.elph.ph.to_pwi(pwi, **pw)

    def save(self, filename):
        """Save driver to file.

        Parameters
        ----------
        filename : str
            Filename for pickled representation of driver.
        """
        interactive = self.interactive
        self.interactive = False

        comms = self.node, self.images

        del self.node
        del self.images

        if not self.sparse:
            elph_comms = self.elph.node, self.elph.images

            del self.elph.node
            del self.elph.images

        elphmod.MPI.Buffer(filename).set(self)

        self.interactive = interactive

        self.node, self.images = comms

        if not self.sparse:
            self.elph.node, self.elph.images = elph_comms

    @staticmethod
    def load(filename):
        """Load driver from file.

        This should only be used in serial runs. Shared memory will be lost.

        Parameters
        ----------
        filename : str
            Filename for pickled representation of driver.

        Returns
        -------
        object
            MD driver for DFPT-based displacements dynamics.
        """
        driver = elphmod.MPI.Buffer(filename).get()

        driver.node, driver.images = elphmod.MPI.shm_split(comm,
            shared_memory=False)

        if not driver.sparse:
            driver.elph.node, driver.elph.images = driver.node, driver.images

        return driver

    def __call__(self, a, r):
        """Interface driver with i-PI.

        Parameters
        ----------
        a : ndarray
            Dummy cell dimensions for variable-cell MD. Our cell is fixed.
        r : ndarray
            Cartesian atomic positions.

        Notes
        -----
        The factor 0.5 converts Rydberg to Hartree units.
        """
        self.u = (r - self.elph.ph.r).ravel()

        E = 0.5 * self.free_energy(show=False)
        F = -0.5 * self.jacobian(show=False)

        virial = np.zeros_like(self.elph.ph.a)
        extras = 'silent'

        return E, F, virial, extras
