import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import h5py


fig, plot_axes = plt.subplots(3, 1, figsize=(3.4, 3.4))
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, wspace=0, hspace=0)

# Parameters
file_nums = [2,8,14]
Bmax = 20
Bmin = 4
lw = 1.4
ms = 4
levels = 6
cadence = 12
filename = lambda s: 'snapshots/snapshots_s{:d}.h5'.format(s)

# Trace center
x_hist = np.array([])
y_hist = np.array([])
for j in range(max(file_nums)):
    f = h5py.File(filename(j+1), 'r')
    x_hist = np.concatenate((x_hist, f['tasks/x'][:,0,0]))
    y_hist = np.concatenate((y_hist, f['tasks/y'][:,0,0]))

# Plot snapshots
for i in range(3):
    with h5py.File(filename(file_nums[i]), 'r') as file:
        # Load data
        x = file['scales/x']['1.0'][:]
        y = file['scales/y']['1.0'][:]
        t = file['scales/sim_time'][0]
        A = file['tasks/A'][0]
        M = file['tasks/M'][0]
        print(M.min(), M.max(), M.mean(), M.std())
        # Plot ellipse
        yy, xx = np.meshgrid(y, x)
        im = plot_axes[i].contourf(xx, yy, M, [0.2,0.5,0.8,1.1], extend='both', cmap='Purples', zorder=1)
        im.cmap.set_under('white')
        im.changed()
        # Plot field
        plot_axes[i].contour(xx, yy, A, levels=np.linspace(Bmin,Bmax,levels), colors=['black'], linewidths=[lw], linestyles=['solid'], zorder=3)
        plot_axes[i].contour(xx, yy, A, levels=np.linspace(-Bmax,-Bmin,levels), colors=['black'], linewidths=[lw], linestyles=['solid'], zorder=3)
        # Plot center
        plot_axes[i].plot(x_hist[:(file_nums[i]-1)*50:cadence],y_hist[:(file_nums[i]-1)*50:cadence], '.', color='C2', ms=ms, zorder=5, lw=lw)
        plot_axes[i].axis([-4,12,-3.9,1.4])
        plot_axes[i].axis('off')

plt.savefig('fig_maglev.pdf')

