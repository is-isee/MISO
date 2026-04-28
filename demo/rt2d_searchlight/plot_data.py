from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pymiso


def parse_boundary_face(conf: pymiso.Conf) -> tuple[str, str]:
    face = conf.searchlight.boundary_face
    axis, side = face.split("_", maxsplit=1)
    if axis not in {"x", "y", "z"} or side not in {"inner", "outer"}:
        raise ValueError(
            "searchlight.boundary_face must be one of "
            "x_inner/x_outer/y_inner/y_outer/z_inner/z_outer"
        )
    return axis, side


def slice_on_boundary(
    values: np.ndarray,
    axis: str,
    side: str,
    d: pymiso.Data,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    full_shape = (d.grid.i_size, d.grid.j_size, d.grid.k_size)
    values_3d = np.reshape(values, full_shape)

    # Select inner (index 0) or outer (index -1) boundary along the chosen axis.
    if side == "inner":
        idx = 0
    elif side == "outer":
        idx = -1
    else:
        raise ValueError("side must be 'inner' or 'outer'")

    if axis == "x":
        # x-boundary face: plane of varying y and z at fixed x.
        return values_3d[idx, :, :], d.y, d.z, "y", "z"
    if axis == "y":
        # y-boundary face: plane of varying x and z at fixed y.
        return values_3d[:, idx, :], d.x, d.z, "x", "z"
    if axis == "z":
        # z-boundary face: plane of varying x and y at fixed z.
        return values_3d[:, :, idx], d.x, d.y, "x", "y"
    raise ValueError("axis must be one of 'x', 'y', or 'z'")


this_dir = Path(__file__).resolve().parent

d = pymiso.Data(data_dir=this_dir / "data")
d.load(0)

fig_dir = this_dir / "figs"
fig_dir.mkdir(exist_ok=True)

axis, side = parse_boundary_face(d.conf)

full_rint = np.reshape(
    d.rint, (d.num_rays, d.grid.i_size, d.grid.j_size, d.grid.k_size)
)
four_pi = 4.0 * np.pi
mean_intensity = np.einsum("r,rijk->ijk", d.weights, full_rint)
radiative_flux_x = four_pi * np.einsum("r,rijk->ijk", d.weights * d.mu_x, full_rint)
radiative_flux_y = four_pi * np.einsum("r,rijk->ijk", d.weights * d.mu_y, full_rint)
radiative_flux_z = four_pi * np.einsum("r,rijk->ijk", d.weights * d.mu_z, full_rint)

plots = [
    ("Mean Intensity", mean_intensity, "inferno"),
    ("Radiative Flux Fx", radiative_flux_x, "coolwarm"),
    ("Radiative Flux Fy", radiative_flux_y, "coolwarm"),
    ("Radiative Flux Fz", radiative_flux_z, "coolwarm"),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

for ax, (title, values, cmap) in zip(axes.flat, plots, strict=True):
    plane, coord_1, coord_2, label_1, label_2 = slice_on_boundary(values, axis, side, d)
    mesh = ax.pcolormesh(coord_1, coord_2, plane.T, shading="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(label_1)
    ax.set_ylabel(label_2)
    ax.set_aspect("equal")
    fig.colorbar(mesh, ax=ax)

fig.savefig(fig_dir / "py_00000000.png", dpi=150)
plt.close(fig)
