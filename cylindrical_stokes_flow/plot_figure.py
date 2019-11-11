"""
Construct figure X from Burns et al (2019)

"""
from dedalus.extras import plot_tools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from dedalus.extras import plot_tools
import h5py
import logging
logger = logging.getLogger(__name__)
plt.style.use('./methods_paper.mplstyle')


# Plot settings
arc_radius = 0.8
frames = [(1,0,'left'),(5,0,'left'),(9,0,'none'),(13,0,'right'),(17,0,'right'),]
outfile = 'fig_stokes_flow.pdf'
dpi = 400
arrow_style = "Simple,tail_width=0.25,head_width=2,head_length=2"
arrow_kw = dict(arrowstyle=arrow_style, color="k")

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
    ax.set_title(r"$t = {:4.1f}$".format(time), fontsize=10, pad=0)
    ax.pcolormesh(theta, r, c1[0,:,:].T, color=colors, rasterized=True)
    ax.annotate(r"$\theta_{{\mathrm{{in}}}} = {:5.1f} \pi$".format(np.abs(phi_in/np.pi)), xytext=(0,0), xy=(0,0), ha='center', va='center', fontsize=9)

    if direction == 'left':
        arrow = patches.FancyArrowPatch((np.pi/2*0.72, arc_radius), (np.pi/2*1.32, arc_radius), connectionstyle="arc3,rad=0.25", **arrow_kw)
        ax.add_patch(arrow)
    elif direction == 'right': 
        arrow = patches.FancyArrowPatch((np.pi/2*1.28, arc_radius), (np.pi/2*0.68, arc_radius), connectionstyle="arc3,rad=-0.25", **arrow_kw)
        ax.add_patch(arrow)
    else:
        pass
    ax.spines['polar'].set_visible(False)

    ## removing the tick marks
    ax.set_xticks([])
    ax.set_yticks([])


fig = plt.figure(figsize=(7, 1.6))
plt.subplots_adjust(left=0, bottom=0, right=1, top=0.9, hspace=0, wspace=0)

# Plotsnapshots
for i, f in enumerate(frames):
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
    ax = fig.add_subplot(1, 5, i+1, projection='polar')
    draw_small_multiple(ax, theta, r, c1, c2, c3, time, phi, direction)
    
# Save figure
plt.tight_layout(pad=1, w_pad=1.5)
fig.savefig(outfile, dpi=dpi)

