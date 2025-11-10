import numpy as np
import matplotlib.pyplot as plt
import pyMISO

data_dir = '../problems/hd_kh_2d/data/'
d = pyMISO.Data(data_dir)

plt.close('all')
fig = plt.figure(figsize=(10, 10))

n0 = 120
n1 = d.n_output
for n in range(n0, n1+1):
    print(n)
    d.load(n)
    ax = fig.add_subplot(111)
    ax.pcolormesh(d.ro.T,vmax = 2.0, vmin = 1.)
    
    if n == n0:
        fig.tight_layout()

    fig.savefig("figs/"+str(n).zfill(8)+".png")
    plt.pause(0.01)
    if n != n1:
        plt.clf()
