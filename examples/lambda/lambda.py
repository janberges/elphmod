#!/usr/bin/env python3

# Copyright (C) 2017-2025 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import elphmod
import numpy as np

nk = nq = 36

el = elphmod.el.Model('TaS2')
mu = elphmod.el.read_Fermi_level('pw.out')
ph = elphmod.ph.Model('ifc', apply_asr_simple=True)
ph.data *= elphmod.misc.Ry ** 2
elph = elphmod.elph.Model('work/TaS2.epmatwp', 'wigner.fmt', el, ph)
elph.data *= elphmod.misc.Ry ** 1.5

q = sorted(elphmod.bravais.irreducibles(nq))
q = 2 * np.pi * np.array(q, dtype=float) / nq

e, U = elphmod.dispersion.dispersion_full_nosym(el.H, nk, vectors=True)
e -= mu

w2, u = elphmod.dispersion.dispersion(ph.D, q, vectors=True)

g2 = elph.sample(q, U=U[..., :1], u=u, squared=True, shared_memory=True)

lamda, wlog, Tc = elphmod.eliashberg.McMillan(nq, e[..., :1], w2, g2,
    mustar=0.0, kT=0.3, f=elphmod.occupations.fermi_dirac)

elphmod.MPI.info('lambda = %g, omega_log = %g meV, Tc = %g K'
    % (lamda, 1e3 * wlog, Tc))
