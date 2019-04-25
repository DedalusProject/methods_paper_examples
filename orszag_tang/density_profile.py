
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import h5py
import publication_settings

matplotlib.rcParams.update(publication_settings.params)

dpi = 300

t_mar, b_mar, l_mar, r_mar = (0.05, 0.25, 0.37, 0.1)
h_main, w_main = (1., 1./publication_settings.golden_mean)

l_sub = 0.65*w_main
b_sub = 0.33*h_main
h_sub = 0.28*h_main
w_sub = 0.28*w_main

h_total = t_mar + h_main + b_mar
w_total = l_mar + w_main + r_mar

width = 3.4
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

# plot
left = l_mar / w_total
bottom = 1 - (t_mar + h_main ) / h_total
width = w_main / w_total
height = h_main / h_total
plot_axis_main = fig.add_axes([left, bottom, width, height])

left = (l_mar + l_sub) / w_total
bottom = 1 - (t_mar + b_sub) / h_total
width = w_sub / w_total
height = h_sub / h_total
plot_axis_sub = fig.add_axes([left, bottom, width, height])

f = h5py.File('snapshots_Re1e4_4096/snapshots_Re1e4_4096_s6.h5')

x = np.array(f['scales/x/1.0'])
rho = np.array(f['tasks/rho'][0,:,1280])

f.close()

plot_axis_main.plot(x,rho,color='k',linewidth=1.5)
plot_axis_main.set_ylim([None,0.5])
plot_axis_main.set_xlabel(r'$x$')
plot_axis_main.set_ylabel(r'$\rho$')

plot_axis_sub.plot(x,rho,color='k',linewidth=0.5)
plot_axis_sub.scatter(x,rho,marker='x',color='k',s=4,linewidth=0.3)
plot_axis_sub.set_xlim([0.69,0.71])
plot_axis_sub.set_xticks([0.69,0.71])
plt.setp(plot_axis_sub.get_xticklabels(), fontsize=8)
plot_axis_sub.yaxis.set_major_locator(MaxNLocator(nbins=3))
plt.setp(plot_axis_sub.get_yticklabels(), fontsize=8)

plt.savefig('OT_line.eps')

