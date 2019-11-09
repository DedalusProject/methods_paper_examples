
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import h5py
from dedalus.extras import plot_tools
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
matplotlib.style.use("methods_paper.mplstyle")


def set_axes_equal(ax, scale=0.5):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = scale*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# Load data
with h5py.File('slices/slices_s40.h5', 'r') as file:
    x = f['scales']['x']['4'][:]
    y = f['scales']['y']['4'][:]
    z = f['scales']['z']['4'][:]
    vo_z = f['tasks']['vorticity-top'][0,:,:,0]
    vo_x = f['tasks']['vorticity-xslice'][0,0,:,:]
    vo_y = f['tasks']['vorticity-yslice'][0,:,0,:]
    pv_z = f['tasks']['PV-top'][0,:,:,0]
    pv_x = f['tasks']['PV-xslice'][0,0,:,:]
    pv_y = f['tasks']['PV-yslice'][0,:,0,:]i

# Vertices
vx = plot_tools.get_1d_vertices(x)
vy = plot_tools.get_1d_vertices(y)
vz = plot_tools.get_1d_vertices(z)

xm, ym = plot_tools.quad_mesh(x, y)

data = [vort,PV]
label = [r'$\omega$',r'$PV$']
cmaps = ['RdBu_r','PuOr']

c_im = []
cbar = []
for i in range(2):
  max = 0.8*np.max(np.abs(data[i]))
  c_im.append(slice_axes[i].pcolormesh(xm, ym, data[i].T, vmin=-max, vmax=max, cmap=cmaps[i]))

  slice_axes[i].axis([-40,40,-20,20])
  if i == 0:
    plt.setp(slice_axes[i].get_xticklabels(), visible=False)
    slice_axes[i].xaxis.set_major_locator(MaxNLocator(nbins=5))
    slice_axes[i].yaxis.set_major_locator(MaxNLocator(nbins=5))
  if i == 1:
    slice_axes[i].set_xlabel(r'$x$')
    slice_axes[i].xaxis.set_major_locator(MaxNLocator(nbins=5))
    slice_axes[i].yaxis.set_major_locator(MaxNLocator(nbins=5,prune='upper'))
  
  slice_axes[i].set_ylabel(r'$y$')

  cbar.append(fig.colorbar(c_im[i], cax=cbar_axes[i], orientation='vertical', ticks=MaxNLocator(nbins=5)))
  cbar_axes[i].yaxis.set_ticks_position('right')
  cbar_axes[i].yaxis.set_label_position('right')
  cbar[i].ax.tick_params(labelsize=8)
  cbar_axes[i].text(3.,0.99,label[i],va='center',ha='center',fontsize=10,transform=cbar_axes[i].transAxes)

plt.savefig('QG.png',dpi=dpi)

