"""Plot the reconnection run (Zenitani 2015, Phys. Plasmas 22, 032114).

As in Fig. 1 of the paper, v_x is shown in the upper half (z > 0) and div v in
the lower half (z < 0), in the axes and units of the paper. A movie of all
outputs is written to figs/reconnection_<plane>.gif for each plane.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

import pymiso

this_dir = Path(__file__).resolve().parent
fig_dir = this_dir / "figs"
fig_dir.mkdir(exist_ok=True)

# MISO axes of the paper's (x, y, z), see src/main.cpp
PLANES = {"zx": "xyz", "xy": "yzx", "yz": "zxy"}

X_RANGE = (0, 130)
Z_RANGE = (-15, 15)


def to_paper(d: pymiso.Data, plane: str) -> dict:
    """Return v_x and div v as 2D arrays [x, z] in the axes of the paper."""
    ax, _, az = PLANES[plane]
    # pymiso drops the axis of size 1; the remaining axes are in the order x, y, z
    transpose = "xyz".index(ax) > "xyz".index(az)

    def arr(a):
        return a.T if transpose else a

    x, z = getattr(d, ax), getattr(d, az)
    vx = arr(getattr(d, "v" + ax))
    vz = arr(getattr(d, "v" + az))
    divv = np.gradient(vx, x, axis=0) + np.gradient(vz, z, axis=1)
    return {"x": x, "z": z, "vx": vx, "divv": divv}


def main():
    for plane in PLANES:
        data_dir = this_dir / f"data_{plane}"
        if not data_dir.exists():
            continue
        d = pymiso.Data(data_dir=data_dir)
        d.load(0)
        f = to_paper(d, plane)
        x, z = f["x"], f["z"]

        fig = plt.figure(figsize=(10, 4.4))
        ax = fig.add_axes((0.07, 0.12, 0.8, 0.8))
        # Upper half: v_x at z > 0. Lower half: div v, mirrored to z < 0.
        im_vx = ax.pcolormesh(
            x, z, f["vx"].T, cmap="jet", vmin=0, vmax=1, shading="nearest"
        )
        im_dv = ax.pcolormesh(
            x, -z, f["divv"].T, cmap="RdBu_r", vmin=-0.2, vmax=0.2, shading="nearest"
        )
        ax.set_xlim(*X_RANGE)
        ax.set_ylim(*Z_RANGE)
        ax.set_aspect(2)  # z is stretched by 2
        # Color bars next to the upper and lower halves
        cax_vx = ax.inset_axes((1.02, 0.53, 0.015, 0.45))
        cax_dv = ax.inset_axes((1.02, 0.02, 0.015, 0.45))
        fig.colorbar(im_vx, cax=cax_vx, label="$v_x$")
        fig.colorbar(im_dv, cax=cax_dv, label=r"$\nabla\cdot v$")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$z$")

        def update(n, d=d, plane=plane, im_vx=im_vx, im_dv=im_dv, ax=ax):
            d.load(n)
            f = to_paper(d, plane)
            im_vx.set_array(f["vx"].T)
            im_dv.set_array(f["divv"].T)
            ax.set_title(f"{plane} plane, t = {d.time.time:.1f}")
            print(plane, n)
            return im_vx, im_dv

        anim = FuncAnimation(fig, update, frames=range(d.n_output + 1))
        anim.save(fig_dir / f"reconnection_{plane}.gif", writer=PillowWriter(fps=5))
        # The last frame as a still image
        fig.savefig(fig_dir / f"reconnection_{plane}.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    main()
