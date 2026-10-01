"""Plot the reconnection run (Zenitani 2015, Phys. Plasmas 22, 032114).

As in Fig. 1 of the paper, v_x is shown in the upper half (z > 0) and div v in
the lower half (z < 0). A movie of all outputs is written to
figs/reconnection.gif, and the last frame to figs/reconnection.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

import pymiso

this_dir = Path(__file__).resolve().parent
fig_dir = this_dir / "figs"
fig_dir.mkdir(exist_ok=True)

d = pymiso.Data(data_dir=this_dir / "data")
x, z = d.x, d.z


def div_v() -> np.ndarray:
    return np.gradient(d.vx, x, axis=0) + np.gradient(d.vz, z, axis=1)


d.load(0)
fig = plt.figure(figsize=(10, 4.4))
ax = fig.add_axes((0.07, 0.12, 0.8, 0.8))
# Upper half: v_x at z > 0. Lower half: div v, mirrored to z < 0.
im_vx = ax.pcolormesh(x, z, d.vx.T, cmap="jet", vmin=0, vmax=1, shading="nearest")
im_dv = ax.pcolormesh(
    x, -z, div_v().T, cmap="RdBu_r", vmin=-0.2, vmax=0.2, shading="nearest"
)
ax.set_xlim(0, 130)
ax.set_ylim(-15, 15)
ax.set_aspect(2)  # z is stretched by 2
# Color bars next to the upper and lower halves
fig.colorbar(im_vx, cax=ax.inset_axes((1.02, 0.53, 0.015, 0.45)), label="$v_x$")
fig.colorbar(
    im_dv, cax=ax.inset_axes((1.02, 0.02, 0.015, 0.45)), label=r"$\nabla\cdot v$"
)
ax.set_xlabel("$x$")
ax.set_ylabel("$z$")


def update(n):
    d.load(n)
    im_vx.set_array(d.vx.T)
    im_dv.set_array(div_v().T)
    ax.set_title(f"t = {d.time.time:.1f}")
    print(n)
    return im_vx, im_dv


anim = FuncAnimation(fig, update, frames=range(d.n_output + 1))
anim.save(fig_dir / "reconnection.gif", writer=PillowWriter(fps=5))
# The last frame as a still image
fig.savefig(fig_dir / "reconnection.png", dpi=120)
plt.close(fig)
