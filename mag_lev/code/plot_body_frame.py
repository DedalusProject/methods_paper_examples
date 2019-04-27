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
fig, ax = plt.subplots()
i_frame = 0
for i in range(4):
    with h5py.File(filename(i+1), 'r') as file:
        x = file['scales/x']['1.0'][:]
        y = file['scales/y']['1.0'][:]
        t = file['scales/sim_time'][:]
        A = file['tasks/A'][:]
        M = file['tasks/M'][:]
        xx, yy = np.meshgrid(x,y)
        for ii in range(len(t)):
            logger.info("file {:s}, frame {:d}".format(filename(i+1),i_frame))
            ax.pcolormesh(xx.T,yy.T,M[ii,:].T-np.min(M[ii,:]),cmap='Purples')
            ax.contour(xx.T,yy.T,A[ii,:].T,
                levels=np.linspace(-Bmax,Bmax,levels),
                colors=['black'],
                linewidths=[lw],
                linestyles=['solid'])
            ax.set_aspect('equal')
            ax.axis('off')
            fig.savefig(figname(i_frame))
            i_frame += 1
            ax.clear()
