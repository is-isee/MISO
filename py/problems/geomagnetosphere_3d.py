import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import pymiso

d = pymiso.Data("../../problems/geomagnetosphere_3d/data")

fig_dir = "figs/geomagnetosphere_3d/"
os.makedirs(fig_dir, exist_ok=True)

plt.close("all")
fig = plt.figure("geomagnetosphere_3d", figsize=(8, 4))

# for n in range(d.n_output, d.n_output + 1):
for n in range(d.n_output + 1):
    d.load(n)
    ax1 = fig.add_subplot(121, aspect="equal")
    ax2 = fig.add_subplot(122, aspect="equal")
    vmin = -4
    vmax = -3
    cmap = "rainbow"
    ax1.pcolormesh(
        d.x,
        d.y,
        np.log10(d.ro[:, :, d.k_size // 2]).T,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    ax2.pcolormesh(
        d.x,
        d.z,
        np.log10(d.ro[:, d.j_size // 2, :]).T,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    circle1 = patches.Circle((0, 0), d.conf.geo_boundary.radius, ec="k", fc="gray")
    circle2 = patches.Circle((0, 0), d.conf.geo_boundary.radius, ec="k", fc="gray")

    ax1.add_patch(circle1)
    ax2.add_patch(circle2)
    ax1.set_xlabel("$x/R_\mathrm{E}$")
    ax2.set_xlabel("$x/R_\mathrm{E}$")

    ax1.set_ylabel("$y/R_\mathrm{E}$")
    ax2.set_ylabel("$z/R_\mathrm{E}$")

    if n == 0:
        fig.tight_layout()

    fig.savefig(fig_dir + "py_" + str(n).zfill(8) + ".png")
    print(n)
    if n != d.n_output:
        plt.clf()
