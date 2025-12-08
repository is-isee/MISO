import os

import matplotlib.pyplot as plt

import pymiso

dx = pymiso.Data(data_dir="data_x/")
dx.load(dx.n_output)

dy = pymiso.Data(data_dir="data_y/")
dy.load(dy.n_output)

dz = pymiso.Data(data_dir="data_z/")
dz.load(dz.n_output)

print("### x-y mean difference in density ###")
print((dx.ro - dy.ro).mean())
print("### x-z mean difference in density ###")
print((dx.ro - dz.ro).mean())

fig = plt.figure(figsize=(14, 4))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.plot(dx.x, dx.ro, marker="o", markerfacecolor="none", label="x", color="blue")
ax1.plot(dy.y, dy.ro, marker="+", markerfacecolor="none", label="y", color="red")
ax1.plot(dz.z, dz.ro, marker="x", markerfacecolor="none", label="z", color="green")

ax2.plot(dx.x, dx.ei, marker="o", markerfacecolor="none", label="x", color="blue")
ax2.plot(dy.y, dy.ei, marker="+", markerfacecolor="none", label="y", color="red")
ax2.plot(dz.z, dz.ei, marker="x", markerfacecolor="none", label="z", color="green")

ax3.plot(dx.x, dx.vx, marker="o", markerfacecolor="none", label="x", color="blue")
ax3.plot(dy.y, dy.vy, marker="+", markerfacecolor="none", label="y", color="red")
ax3.plot(dz.z, dz.vz, marker="x", markerfacecolor="none", label="z", color="green")

ax1.set_title("Density")
ax2.set_title("Internal Energy")
ax3.set_title("Velocity")

fig.tight_layout()

fig_dir = "figs/"
os.makedirs(fig_dir, exist_ok=True)
fig.savefig(fig_dir + "hd_shock_tube_1d.png")
plt.close("all")
