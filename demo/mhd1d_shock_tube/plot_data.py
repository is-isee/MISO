import os

import matplotlib.pyplot as plt

import pymiso

problems_dir = "../../problems/mhd_shock_tube_1d"
data_dir = problems_dir + "/data_x/"
dx = pymiso.Data(data_dir)
dx.load(dx.n_output)

data_dir = problems_dir + "/data_y/"
dy = pymiso.Data(data_dir)
dy.load(dy.n_output)

data_dir = problems_dir + "/data_z/"
dz = pymiso.Data(data_dir)
dz.load(dz.n_output)

print("### x-y mean difference in density ###")
print((dx.ro - dy.ro).mean())
print("### x-z mean difference in density ###")
print((dx.ro - dz.ro).mean())

plt.close("all")
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)

ax1.plot(dx.x, dx.ro, marker="o", markerfacecolor="none", label="x", color="blue")
ax1.plot(dy.y, dy.ro, marker="o", markerfacecolor="none", label="y", color="red")
ax1.plot(dz.z, dz.ro, marker="o", markerfacecolor="none", label="z", color="green")

ax2.plot(dx.x, dx.ei, marker="o", markerfacecolor="none", label="x", color="blue")
ax2.plot(dy.y, dy.ei, marker="o", markerfacecolor="none", label="y", color="red")
ax2.plot(dz.z, dz.ei, marker="o", markerfacecolor="none", label="z", color="green")

ax3.plot(dx.x, dx.vx, marker="o", markerfacecolor="none", label="x", color="blue")
ax3.plot(dy.y, dy.vy, marker="o", markerfacecolor="none", label="y", color="red")
ax3.plot(dz.z, dz.vz, marker="o", markerfacecolor="none", label="z", color="green")

ax4.plot(dx.x, dx.vy, marker="o", markerfacecolor="none", label="x", color="blue")
ax4.plot(dy.y, dy.vz, marker="o", markerfacecolor="none", label="y", color="red")
ax4.plot(dz.z, dz.vx, marker="o", markerfacecolor="none", label="z", color="green")

ax5.plot(dx.x, dx.by, marker="o", markerfacecolor="none", label="x", color="blue")
ax5.plot(dy.y, dy.bz, marker="o", markerfacecolor="none", label="y", color="red")
ax5.plot(dz.z, dz.bx, marker="o", markerfacecolor="none", label="z", color="green")


# calculate pressure
for d in [dx, dy, dz]:
    d.pr = d.ro * d.ei * (d.conf.eos.gm - 1)

ax6.plot(dx.x, dx.pr, marker="o", markerfacecolor="none", label="x", color="blue")
ax6.plot(dy.y, dy.pr, marker="o", markerfacecolor="none", label="y", color="red")
ax6.plot(dz.z, dz.pr, marker="o", markerfacecolor="none", label="z", color="green")

ax1.set_title("Density")
ax2.set_title("Internal Energy")
ax3.set_title("Axial Velocity")
ax4.set_title("Transverse Velocity")
ax5.set_title("Transverse Magnetic Field")
ax6.set_title("Pressure")

fig.tight_layout()

fig_dir = "figs/"
os.makedirs(fig_dir, exist_ok=True)
fig.savefig(fig_dir + "mhd_shock_tube_1d.png")
