from pathlib import Path

import matplotlib.pyplot as plt

import pymiso

this_dir = Path(__file__).resolve().parent

d = pymiso.Data(data_dir=this_dir / "data")

fig_dir = Path(this_dir / "figs")
fig_dir.mkdir(exist_ok=True)

fig = plt.figure(figsize=(5, 5))

for n in range(d.n_output + 1):
    d.load(n)
    ax = fig.add_subplot(111, aspect="equal")
    ax.pcolormesh(d.x, d.y, d.ro.T, shading="auto", vmin=0.9, vmax=2.1, cmap="inferno")

    if n == 0:
        fig.tight_layout()

    fig.savefig(fig_dir / f"py_{n:08d}.png")
    print(n)
    if n != d.n_output:
        plt.clf()

plt.close("all")
