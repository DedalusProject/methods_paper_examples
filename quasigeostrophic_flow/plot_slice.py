
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import h5py
from dedalus.extras import plot_tools


dpi = 600

t_mar, b_mar, l_mar, r_mar = (0.1, 0.35, 0.45, 0.3)
h_slice, w_slice = (1., 2.)
h_pad = 0.05
w_pad = 0.05

h_cbar, w_cbar = (h_slice, 0.05)

h_total = t_mar + 2*h_slice + h_pad + b_mar
w_total = l_mar + w_slice + w_pad + w_cbar + r_mar

width = 3.4
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

# slices
slice_axes = []
for i in range(2):
    left = l_mar / w_total
    bottom = 1 - (t_mar + i*(h_pad + h_slice) + h_slice ) / h_total
    width = w_slice / w_total
    height = h_slice / h_total
    slice_axes.append(fig.add_axes([left, bottom, width, height]))

# cbars
cbar_axes = []
for i in range(2):
    left = (l_mar + w_slice + w_pad) / w_total
    bottom = 1 - (t_mar + i*(h_pad+h_slice) + h_slice ) / h_total
    width = w_cbar / w_total
    height = h_cbar / h_total
    cbar_axes.append(fig.add_axes([left, bottom, width, height]))

f = h5py.File('slices/slices_s40.h5')

for a in f['scales/x']: print(a)
x = np.array(f['scales/x/4'])
y = np.array(f['scales/y/4'])
vort = np.array(f['tasks/vorticity-top'][0,:,:,0])
PV   = np.array(f['tasks/PV-top'][0,:,:,0])

f.close()

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

