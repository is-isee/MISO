import os

import matplotlib.pyplot as plt

import pyMISO

d = pyMISO.Data("../../problems/mhd_vortex_2d/data")

fig_dir = "figs/mhd_vortex_2d/"
os.makedirs(fig_dir, exist_ok=True)


plt.close("all")
fig = plt.figure("mhd_vortex_2d", figsize=(5, 5))

for n in range(d.n_output + 1):
    d.load(n)
    ax = fig.add_subplot(111, aspect="equal")
    ax.pcolormesh(d.x, d.y, d.ro.T, shading="auto", vmin=0.2, vmax=2.5, cmap="inferno")

    if n == 0:
        fig.tight_layout()

    fig.savefig(fig_dir + "py_" + str(n).zfill(8) + ".png")
    plt.pause(0.01)
    print(n)
    if n != d.n_output:
        plt.clf()
