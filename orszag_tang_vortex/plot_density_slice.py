
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
from dedalus.extras import plot_tools
plt.style.use('./methods_paper.mplstyle')


# Load data
f = h5py.File('snapshots_Re1e4_4096/snapshots_Re1e4_4096_s6.h5')
x = np.array(f['scales/x/1.0'])
y = np.array(f['scales/y/1.0'])
rho = np.array(f['tasks/rho'][0])
f.close()

# Plot slice
xm, ym = plot_tools.quad_mesh(x, y)
fig = plt.figure(figsize=(3.4, 3.8))
slice_axis = fig.add_subplot(111)
im = slice_axis.pcolormesh(xm, ym, rho.T, cmap='gray_r')
slice_axis.axis([0,1,0,1])
slice_axis.set_xlabel(r'$x$')
slice_axis.set_ylabel(r'$y$')

# Add colorbar
divider = make_axes_locatable(slice_axis)
cax = divider.append_axes("top", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
cax.xaxis.set_ticks_position('top')
cax.xaxis.set_label_position('top')
cax.tick_params(labelsize=8)
cax.set_title(r'$\rho$')
slice_axis.set_aspect('equal')

# Save
plt.tight_layout(pad=0.2)
plt.savefig('fig_OT_image.jpg', dpi=400)


