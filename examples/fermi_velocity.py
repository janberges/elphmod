#/usr/bin/env python3

import elphmod
import numpy as np

comm = elphmod.MPI.comm
info = elphmod.MPI.info

mu = -0.1665
nk = 120
points = 500

cmap = elphmod.plot.colormap(
    (0, elphmod.plot.Color(255, 255, 255)),
    (1, elphmod.plot.Color(  0,   0,   0)),
    )

el = elphmod.el.Model('data/NbSe2_hr.dat')

e = elphmod.dispersion.dispersion_full(el.H, nk)[:, :, 0] - mu

kxmax, kymax, kx, ky, e = elphmod.plot.toBZ(e,
    points=points, return_k=True, outside=np.nan)

dedky, dedkx = np.gradient(e, ky, kx)
dedk = np.sqrt(dedkx ** 2 + dedky ** 2)

image = elphmod.plot.color(dedk, cmap)

if comm.rank == 0:
    elphmod.plot.save('fermi_velocity.png', image)

info("Min./max./mean number of k-points for meV resolution:")

FS = np.where(np.logical_and(~np.isnan(dedk), abs(e) < 0.1))

for v in dedk[FS].min(), dedk[FS].max(), np.average(dedk[FS]):
    info(int(round(2 * kymax * v / 1e-3)))
