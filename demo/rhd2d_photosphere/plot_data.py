from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pymiso

this_dir = Path(__file__).resolve().parent


def read_table(path):
    """Read an r2d2plus-table (see input_data/README.md)."""
    with open(path, "rb") as f:
        head = f.read(65536).decode("utf-8", errors="replace")
    header, shapes = {}, {}
    for line in head.splitlines()[1:]:
        if line == "end":
            break
        key, _, rest = line.partition(" ")
        if key == "array":
            name, *shape = rest.split()
            shapes[name] = tuple(int(n) for n in shape)
        elif key and key != "meta":
            header[key] = rest
    arrays, offset = {}, int(header["binary_offset"])
    for name, shape in shapes.items():
        count = int(np.prod(shape))
        arrays[name] = np.fromfile(path, dtype="<f8", count=count, offset=offset)
        arrays[name] = arrays[name].reshape(shape)
        offset += 8 * count
    return arrays


def temperature(table, ro, ei):
    """T from the EOS table, bilinear in (ln rho, ln e) as in MISO."""
    log_te = np.log(np.maximum(table["te"], 1e-20))
    x = np.log(ro)
    y = np.log(ei)
    fi = (x - table["ro"][0]) / (table["ro"][1] - table["ro"][0])
    fj = (y - table["en"][0]) / (table["en"][1] - table["en"][0])
    n0, n1 = log_te.shape
    fi = np.clip(fi, 0, n0 - 1)
    fj = np.clip(fj, 0, n1 - 1)
    i = np.clip(fi.astype(int), 0, n0 - 2)
    j = np.clip(fj.astype(int), 0, n1 - 2)
    t, u = fi - i, fj - j
    return np.exp(
        (1 - t) * (1 - u) * log_te[i, j]
        + (1 - t) * u * log_te[i, j + 1]
        + t * (1 - u) * log_te[i + 1, j]
        + t * u * log_te[i + 1, j + 1]
    )


d = pymiso.Data(data_dir=this_dir / "data")
table = read_table(this_dir / "../../input_data/eos_table_roen_blend.tbl")

fig_dir = this_dir / "figs"
fig_dir.mkdir(exist_ok=True)

# Temperature and vertical velocity at the last output
d.load(d.n_output)
x, z = d.x / 1e8, d.z / 1e8
te = temperature(table, d.ro, d.ei)
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
im = axs[0].pcolormesh(x, z, te.T, shading="auto", cmap="inferno")
fig.colorbar(im, ax=axs[0], label="T [K]")
vmax = np.abs(d.vz).max() / 1e5
im = axs[1].pcolormesh(
    x, z, d.vz.T / 1e5, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax
)
fig.colorbar(im, ax=axs[1], label="$v_z$ [km/s]")
for ax in axs:
    ax.set_aspect("equal")
    ax.set_ylabel("z [Mm]")
axs[0].set_title(f"t = {d.time.time:.0f} s")
axs[1].set_xlabel("x [Mm]")
fig.tight_layout()
fig.savefig(fig_dir / "photosphere.png")

# Effective temperature from the radiative flux at the top
t, teff = np.loadtxt(this_dir / "data" / "teff.txt", unpack=True)
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(t / 60, teff)
ax.axhline(5772, color="gray", linestyle="--", label="Sun (5772 K)")
ax.set_xlabel("t [min]")
ax.set_ylabel("$T_\\mathrm{eff}$ [K]")
ax.legend()
fig.tight_layout()
fig.savefig(fig_dir / "teff.png")

# rms of vz near z = 0
k0 = np.argmin(np.abs(z))
print(
    f"T_eff = {teff[-1]:.0f} K at t = {t[-1]:.0f} s; at t = {d.time.time:.0f} s, "
    f"rms vz at z = {z[k0]:.2f} Mm: {np.sqrt((d.vz[:, k0] ** 2).mean()) / 1e5:.2f} km/s"
)
