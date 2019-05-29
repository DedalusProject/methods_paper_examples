
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import h5py
import publication_settings
from dedalus.extras import plot_tools

matplotlib.rcParams.update(publication_settings.params)

dpi = 600

t_mar, b_mar, l_mar, r_mar = (0.18, 0.18, 0.18, 0.05)
h_slice, w_slice = (1., 1.)
h_pad = 0.05

h_cbar, w_cbar = (0.05, w_slice)

h_total = t_mar + h_pad + h_cbar + h_slice + b_mar
w_total = l_mar + w_slice + r_mar

width = 3.4
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

# slices
left = l_mar / w_total
bottom = 1 - (t_mar + h_cbar + h_pad + h_slice ) / h_total
width = w_slice / w_total
height = h_slice / h_total
slice_axis = fig.add_axes([left, bottom, width, height])

# cbars
left = l_mar / w_total
bottom = 1 - (t_mar + h_cbar ) / h_total
width = w_cbar / w_total
height = h_cbar / h_total
cbar_axis = fig.add_axes([left, bottom, width, height])

f = h5py.File('snapshots_Re1e4_4096/snapshots_Re1e4_4096_s6.h5')

x = np.array(f['scales/x/1.0'])
y = np.array(f['scales/y/1.0'])
rho = np.array(f['tasks/rho'][0])

f.close()

xm, ym = plot_tools.quad_mesh(x, y)

c_im = slice_axis.pcolormesh(xm, ym, rho.T, cmap='gray_r')

slice_axis.axis([0,1,0,1])
slice_axis.set_xlabel(r'$x$')
slice_axis.set_ylabel(r'$y$')

cbar = fig.colorbar(c_im, cax=cbar_axis, orientation='horizontal', ticks=MaxNLocator(nbins=8))
cbar_axis.xaxis.set_ticks_position('top')
cbar_axis.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=8)
cbar_axis.text(0.5,4.,r'$\rho$',va='center',ha='center',fontsize=10,transform=cbar_axis.transAxes)

plt.savefig('OT.png',dpi=dpi)

