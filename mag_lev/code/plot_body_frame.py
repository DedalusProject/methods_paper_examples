import matplotlib.pyplot as plt
import numpy as np
import h5py

import logging
logger = logging.getLogger(__name__)

Bmax = 20

filename = lambda s: 'snapshots/snapshots_s{:d}.h5'.format(s)
figname =  lambda s: 'frames_{:06d}.png'.format(s)
lw=4.5
levels=12

t = np.array([])
x_hist = np.array([])
y_hist = np.array([])

fig, ax = plt.subplots()
i_frame = 0
for i in range(7):
    with h5py.File(filename(i+1), 'r') as file:
        x_hist = np.concatenate((x_hist, file['tasks/x'][:,0,0]))
        y_hist = np.concatenate((y_hist, file['tasks/y'][:,0,0]))

        x = file['scales/x']['4'][:]
        y = file['scales/y']['4'][:]
        t = file['scales/sim_time'][:]
        A = file['tasks/A'][:]
        M = file['tasks/M'][:]
        yy, xx = np.meshgrid(y,x)
        for ii in range(len(t)):
            logger.info("file {:s}, frame {:d}".format(filename(i+1),i_frame))
            ax.pcolormesh(xx,yy,M[ii,:]-np.min(M[ii,:]),cmap='Purples')
            ax.contour(xx,yy,A[ii,:],
                levels=np.linspace(-Bmax,Bmax,levels),
                colors=['black'],
                linewidths=[lw],
                linestyles=['solid'])
            ax.plot(x_hist[:i_frame], y_hist[:i_frame], color='slategrey')
            ax.set_aspect('equal')
            ax.axis('off')
            fig.savefig(figname(i_frame))
            i_frame += 1
            ax.clear()
