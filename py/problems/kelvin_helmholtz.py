import os

import matplotlib.pyplot as plt

import pyMISO

d = pyMISO.Data("../../problems/kelvin_helmholtz/data")

fig_dir = "figs/kelvin_helmholtz/"
os.makedirs(fig_dir, exist_ok=True)


plt.close("all")
fig = plt.figure("kelvin_helmholtz", figsize=(5, 5))

for n in range(d.n_output + 1):
    d.load(n)
    ax = fig.add_subplot(111, aspect="equal")
    ax.pcolormesh(d.x, d.y, d.ro.T, shading="auto", vmin=0.9, vmax=2.1, cmap="inferno")

    if n == 0:
        fig.tight_layout()

    fig.savefig(fig_dir + "py_" + str(n).zfill(8) + ".png")
    print(n)
    if n != d.n_output:
        plt.clf()
