import numpy as np
import matplotlib.pyplot as plt
import pyrmhdlab

data_dir = '../problems/hd_shock_tube_1d/data_x/'
dx = pyrmhdlab.Data(data_dir)
dx.load(dx.n_output)

data_dir = '../problems/hd_shock_tube_1d/data_y/'
dy = pyrmhdlab.Data(data_dir)
dy.load(dy.n_output)

data_dir = '../problems/hd_shock_tube_1d/data_z/'
dz = pyrmhdlab.Data(data_dir)
dz.load(dz.n_output)

print('### x-y mean difference in density ###')
print((dx.ro - dy.ro).mean())
print('### x-z mean difference in density ###')
print((dx.ro - dz.ro).mean())

plt.close('all')
fig = plt.figure(figsize=(14, 4))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.plot(dx.ro, marker='o', markerfacecolor='none', label='x', color='blue')
ax1.plot(dy.ro, marker='o', markerfacecolor='none', label='y', color='red')
ax1.plot(dz.ro, marker='o', markerfacecolor='none', label='z', color='green')

ax2.plot(dx.ei, marker='o', markerfacecolor='none', label='x', color='blue')
ax2.plot(dy.ei, marker='o', markerfacecolor='none', label='y', color='red')
ax2.plot(dz.ei, marker='o', markerfacecolor='none', label='z', color='green')

ax3.plot(dx.vx, marker='o', markerfacecolor='none', label='x', color='blue')
ax3.plot(dy.vy, marker='o', markerfacecolor='none', label='y', color='red')
ax3.plot(dz.vz, marker='o', markerfacecolor='none', label='z', color='green')

ax1.set_title('Density')
ax2.set_title('Internal Energy')
ax3.set_title('Velocity')

fig.tight_layout()