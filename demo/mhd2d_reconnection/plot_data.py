"""Plot the reconnection run (Zenitani 2015, Phys. Plasmas 22, 032114).

Figures (in the units and axes of the paper):
- reconnection_<plane>_<n>.png: v_x (upper half) and div v (lower half),
  as in Fig. 1 of the paper, and the density, as in Fig. 2
- reconnection_cut.png: profiles along z = 0 at the last output (Fig. 4),
  for the zx, xy, and yz planes
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pymiso

this_dir = Path(__file__).resolve().parent
fig_dir = this_dir / "figs"
fig_dir.mkdir(exist_ok=True)

# MISO axes of the paper's (x, y, z), see src/main.cpp
PLANES = {"zx": "xyz", "xy": "yzx", "yz": "zxy"}


def to_paper(d: pymiso.Data, plane: str) -> dict:
    """Return the fields as 2D arrays [x, z] in the axes and units of the paper."""
    ax, _, az = PLANES[plane]
    # pymiso drops the axis of size 1; the remaining axes are in the order x, y, z
    transpose = "xyz".index(ax) > "xyz".index(az)

    def arr(a):
        return a.T if transpose else a

    b_unit = np.sqrt(4 * np.pi)
    f = {
        "x": getattr(d, ax),
        "z": getattr(d, az),
        "ro": arr(d.ro),
        "pr": arr(d.ro * d.ei * (d.conf.eos.gm - 1)),
        "vx": arr(getattr(d, "v" + ax)),
        "vz": arr(getattr(d, "v" + az)),
        "bx": arr(getattr(d, "b" + ax)) / b_unit,
        "bz": arr(getattr(d, "b" + az)) / b_unit,
    }
    dvx = np.gradient(f["vx"], f["x"], axis=0)
    dvz = np.gradient(f["vz"], f["z"], axis=1)
    f["divv"] = dvx + dvz
    return f


def mirror(a: np.ndarray, odd: bool = False) -> np.ndarray:
    """Extend [x, z] (z >= 0) to z < 0 by the symmetry of the problem"""
    return np.concatenate([(-a if odd else a)[:, ::-1], a], axis=1)


def plot_2d(f: dict, plane: str, n: int, time: float):
    x, z = f["x"], f["z"]
    zz = np.concatenate([-z[::-1], z])
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), layout="constrained")

    # Fig. 1: v_x (z > 0) and div v (z < 0)
    ax = axs[0]
    upper = np.where(zz[None, :] > 0, mirror(f["vx"]), np.nan)
    lower = np.where(zz[None, :] < 0, mirror(f["divv"]), np.nan)
    im1 = ax.pcolormesh(x, zz, upper.T, cmap="jet", vmin=0, vmax=1, shading="auto")
    im2 = ax.pcolormesh(
        x, zz, lower.T, cmap="RdBu_r", vmin=-0.2, vmax=0.2, shading="auto"
    )
    fig.colorbar(im1, ax=ax, label="$v_x$ ($z > 0$)", pad=0.01)
    fig.colorbar(im2, ax=ax, label=r"$\nabla\cdot v$ ($z < 0$)", pad=0.01)
    ax.set_title(f"{plane} plane, t = {time:.1f}")

    # Fig. 2: density
    ax = axs[1]
    im = ax.pcolormesh(x, zz, mirror(f["ro"]).T, cmap="jet", shading="auto")
    fig.colorbar(im, ax=ax, label=r"$\rho$", pad=0.01)

    for ax in axs:
        ax.set_aspect("equal")
        ax.set_xlim(0, x[-1])
        ax.set_ylim(-30, 30)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$z$")
    fig.savefig(fig_dir / f"reconnection_{plane}_{n:08d}.png", dpi=120)
    plt.close(fig)


def main():
    last = {}
    for plane in PLANES:
        data_dir = this_dir / f"data_{plane}"
        if not data_dir.exists():
            continue
        d = pymiso.Data(data_dir=data_dir)
        for n in range(d.n_output + 1):
            d.load(n)
            f = to_paper(d, plane)
            plot_2d(f, plane, n, d.time.time)
            print(plane, n)
        last[plane] = f

    if not last:
        return

    # Plane dependence: all planes must give the same result
    ref = last.get("zx")
    for plane, f in last.items():
        if plane != "zx" and ref is not None:
            diff = max(np.abs(f[v] - ref[v]).max() for v in ("ro", "vx", "vz", "bx"))
            print(f"max difference {plane} - zx: {diff:.3e}")

    # Fig. 4: cut along z = 0
    fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    for (plane, f), ls in zip(last.items(), ["-", "--", ":"], strict=False):
        for ax, v, label in zip(
            axs,
            ["ro", "pr", "vx", "bz"],
            [r"$\rho$", "$p$", "$v_x$", "$B_z$"],
            strict=True,
        ):
            ax.plot(f["x"], f[v][:, 0], ls, label=plane)
            ax.set_ylabel(label)
    axs[0].legend()
    axs[-1].set_xlabel("$x$ ($z = 0$)")
    fig.tight_layout()
    fig.savefig(fig_dir / "reconnection_cut.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
