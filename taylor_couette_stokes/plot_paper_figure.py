"""
Construct figure X from Burns et al (2019)

"""
from dedalus.extras import plot_tools
import numpy as np
import matplotlib.pyplot as plt
from dedalus.extras import plot_tools
import h5py
import logging
logger = logging.getLogger(__name__)

# Plot settings
dpi = 800
arc_radius = 0.8

frames = [(1,0,'left'),(5,0,'left'),(9,0,'none'),(13,0,'right'),(16,-1,'right'),]
outfile = 'stokes_flow_figure.png'

def draw_small_multiple(ax, theta, r, c1, c2, c3, time, phi, direction):
    c1_mod = c1[index]
    c1_mod[c1[index]<0] = 0
    c1_mod[c1[index]>1] = 1
    c2_mod = c2[index]
    c2_mod[c2[index]<0] = 0
    c2_mod[c2[index]>1] = 1
    c3_mod = c3[index]
    c3_mod[c3[index]<0] = 0
    c3_mod[c3[index]>1] = 1
    phi_in = phi[index,0,0]
    colors = tuple(np.array([c1_mod.T.flatten(),c2_mod.T.flatten(),c3_mod.T.flatten()]).transpose().tolist())
    ax.set_title("t = {:4.1f}".format(time), fontsize=24)
    ax.pcolormesh(theta,r,c1[0,:,:].T,color=colors)
    ax.annotate(r"$\phi = {:5.1f} \pi$".format(phi_in/np.pi), xytext=(0,0), xy=(0,0), ha='center',fontsize=20)

    if direction == 'left':
        ax.annotate("", xy=(3*np.pi/4., arc_radius), xytext=(np.pi/2,arc_radius),
                    arrowprops=dict(arrowstyle='->',
                                    connectionstyle='arc3,rad={}'.format(0.3),
                                    linewidth=4))
    elif direction == 'right':
        ax.annotate("", xy=(np.pi/2., arc_radius), xytext=(np.pi/4,arc_radius),
                    arrowprops=dict(arrowstyle='<-',
                                    connectionstyle='arc3,rad={}'.format(0.3),
                                    linewidth=4))
    else:
        pass
    ax.spines['polar'].set_visible(False)

    ## removing the tick marks
    ax.set_xticks([])
    ax.set_yticks([])


fig = plt.figure(figsize=(30,6))


# Plot writes
for i,f in enumerate(frames):
    fn, index, direction = f
    print("Plotting frame {:d} of {:d}".format(i+1,len(frames)))

    filename = "snapshots/snapshots_s{:d}.h5".format(fn)
    logger.info("Ploting index {:d} from file {:s} with direction {:s}".format(index, filename, direction))
    file = h5py.File(filename,"r")
    r = file['scales/r/1.0']
    theta = file['scales/θ/1.0']

    theta,r = plot_tools.quad_mesh(theta,r)
    c1 = file['tasks/c1']
    c2 = file['tasks/c2']
    c3 = file['tasks/c3']
    phi = file['tasks/left(φ)']
    time = file['scales/sim_time'][index]
    ax = fig.add_subplot(1,5,i+1,projection='polar')
    draw_small_multiple(ax, theta, r, c1, c2, c3, time, phi, direction)
    # Save figure
fig.savefig(outfile, dpi=dpi)

