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

t_mar, b_mar, l_mar, r_mar = (0.05, 0.05, 0.05, 0.05)
h_slice, w_slice = (1., 2.)

h_total = t_mar + h_slice + b_mar
w_total = l_mar + w_slice + r_mar

width = 3.4
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

# axis
left = l_mar / w_total
bottom = 1 - (t_mar + h_slice ) / h_total
width = w_slice / w_total
height = h_slice / h_total
axis = fig.add_axes([left, bottom, width, height])

Bmax = 20
Bmin = 4

filename = lambda s: 'snapshots/snapshots_s{:d}.h5'.format(s)
figname =  lambda s: 'frames/frames_{:06d}.png'.format(s)
lw=2
levels=6

for i in range(rank,14,size):
    x_hist = np.array([])
    y_hist = np.array([])
    t = np.array([])
    for j in range(i+1):
        f = h5py.File(filename(j+1), 'r')
        x_hist = np.concatenate((x_hist, f['tasks/x'][:,0,0]))
        y_hist = np.concatenate((y_hist, f['tasks/y'][:,0,0]))
    with h5py.File(filename(i+1), 'r') as file:

        x = file['scales/x']['1.0'][:]
        y = file['scales/y']['1.0'][:]
        t = file['scales/sim_time'][:]
        A = file['tasks/A'][:]
        M = file['tasks/M'][:]
        yy, xx = np.meshgrid(y,x)
        for ii in range(len(t)):
            i_frame = ii + i*50
            axis.pcolormesh(xx,yy,M[ii,:]-np.min(M[ii,:]),cmap='Purples')
            axis.contour(xx,yy,A[ii,:],
                  levels=np.linspace(Bmin,Bmax,levels),
                  colors=['black'],
                  linewidths=[lw],
                  linestyles=['solid'])
            axis.contour(xx,yy,A[ii,:],
                  levels=np.linspace(-Bmax,-Bmin,levels),
                  colors=['black'],
                  linewidths=[lw],
                  linestyles=['solid'])
            axis.plot(x_hist[:i_frame], y_hist[:i_frame], color='slategrey', linewidth=lw)
            axis.axis('off')
            fig.savefig(figname(i_frame),dpi=300)
            axis.clear()
