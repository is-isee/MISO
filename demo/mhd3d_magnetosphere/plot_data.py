from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import pymiso

this_dir = Path(__file__).resolve().parent

d = pymiso.Data(data_dir=this_dir / "data")

fig_dir = Path(this_dir / "figs")
fig_dir.mkdir(exist_ok=True)

plt.close("all")
fig = plt.figure("mhd3d_magnetosphere", figsize=(8, 4))

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

    circle1 = patches.Circle((0, 0), d.conf.magnetosphere.radius, ec="k", fc="gray")
    circle2 = patches.Circle((0, 0), d.conf.magnetosphere.radius, ec="k", fc="gray")

    ax1.add_patch(circle1)
    ax2.add_patch(circle2)
    ax1.set_xlabel("$x/R_\mathrm{E}$")
    ax2.set_xlabel("$x/R_\mathrm{E}$")

    ax1.set_ylabel("$y/R_\mathrm{E}$")
    ax2.set_ylabel("$z/R_\mathrm{E}$")

    if n == 0:
        fig.tight_layout()

    fig.savefig(fig_dir / f"py_{n:08d}.png")
    print(n)
    if n != d.n_output:
        plt.clf()
