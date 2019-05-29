import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import h5py
from mpi4py import MPI
import publication_settings

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

matplotlib.rcParams.update(publication_settings.params)

t_mar, b_mar, l_mar, r_mar = (0.01, 0.01, 0.01, 0.01)
h_slice, w_slice = (0.3125, 1)
h_pad = 0.03

h_total = t_mar + 3*h_slice + 2*h_pad + b_mar
w_total = l_mar + w_slice + r_mar

width = 3.4
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

# axis
plot_axes = []
for i in range(3):
    left = l_mar / w_total
    bottom = 1 - (t_mar + h_slice + i*(h_slice+h_pad) ) / h_total
    width = w_slice / w_total
    height = h_slice / h_total
    plot_axes.append(fig.add_axes([left, bottom, width, height]))

Bmax = 20
Bmin = 4

filename = lambda s: 'snapshots/snapshots_s{:d}.h5'.format(s)
lw=2
levels=6

x_hist = np.array([])
y_hist = np.array([])
for j in range(14):
    f = h5py.File(filename(j+1), 'r')
    x_hist = np.concatenate((x_hist, f['tasks/x'][:,0,0]))
    y_hist = np.concatenate((y_hist, f['tasks/y'][:,0,0]))

file_nums = [2,8,14]

for i in range(3):
    with h5py.File(filename(file_nums[i]), 'r') as file:

        x = file['scales/x']['1.0'][:]
        y = file['scales/y']['1.0'][:]
        t = file['scales/sim_time'][0]
        A = file['tasks/A'][0]
        M = file['tasks/M'][0]
        yy, xx = np.meshgrid(y,x)
        im = plot_axes[i].pcolormesh(xx,yy,M-np.min(M),cmap='Purples')
        plot_axes[i].contour(xx,yy,A,
                  levels=np.linspace(Bmin,Bmax,levels),
                  colors=['black'],
                  linewidths=[lw],
                  linestyles=['solid'])
        plot_axes[i].contour(xx,yy,A,
                  levels=np.linspace(-Bmax,-Bmin,levels),
                  colors=['black'],
                  linewidths=[lw],
                  linestyles=['solid'])
        plot_axes[i].scatter(x_hist[:(file_nums[i]-1)*50],y_hist[:(file_nums[i]-1)*50], color='slategrey', edgecolor='slategrey', marker='o', zorder=4, s=0.3)
        plot_axes[i].axis([-4,12,-4,1])
        plot_axes[i].axis('off')

plt.savefig('mag_lev.png', dpi=600)

