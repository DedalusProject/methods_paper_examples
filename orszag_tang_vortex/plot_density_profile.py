
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import h5py
plt.style.use('./methods_paper.mplstyle')


# Load data
f = h5py.File('snapshots_Re1e4_4096/snapshots_Re1e4_4096_s6.h5')
x = np.array(f['scales/x/1.0'])
rho = np.array(f['tasks/rho'][0,:,1280])
f.close()

# Plot profile
fig = plt.figure(figsize=(3.4, 2.1))
plot_axis_main = fig.add_subplot(111)
plot_axis_main.plot(x,rho,color='k',linewidth=1.5)
plot_axis_main.set_xlim((0, 1.0))
plot_axis_main.set_ylim([None,0.5])
plot_axis_main.set_xlabel(r'$x$')
plot_axis_main.set_ylabel(r'$\rho$')

# Subplot
plot_axis_sub = fig.add_axes([0.68, 0.65, 0.24, 0.28])
plot_axis_sub.plot(x,rho,color='k',linewidth=0.5)
plot_axis_sub.scatter(x,rho,marker='x',color='k',s=4,linewidth=0.3)
plot_axis_sub.set_xlim([0.69,0.71])
plt.setp(plot_axis_sub.get_xticklabels(), fontsize=8)
plt.setp(plot_axis_sub.get_yticklabels(), fontsize=8)

plt.tight_layout(pad=0.5)
plt.savefig('fig_OT_line.pdf')

