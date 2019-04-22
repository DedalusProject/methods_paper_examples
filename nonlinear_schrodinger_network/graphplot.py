"""
Plot output from NLS simulation.

Usage:
    plot_snapshots.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil
import os
import pathlib
import h5py


# Load graph
with np.load('graph.npz') as graph:
    edges = graph['edges']
    verts = graph['vertices']

# Variable name definitions as functions of edge number
str_edge = lambda ne: f"{edges[ne][0]}_{edges[ne][1]}"
str_u = lambda ne: f"u_{str_edge(ne)}"
str_ux = lambda ne: f"ux_{str_edge(ne)}"

# Plot edges
def edge_plot(ulist, tlist, title, fname):
    u_array = np.array(ulist)
    u_array = np.abs(u_array)**2
    t_array = np.array(tlist)
    xmesh, ymesh = quad_mesh(x=x, y=t_array)
    plt.figure()
    plt.pcolormesh(xmesh, ymesh, u_array, cmap='RdBu_r')
    plt.axis(pad_limits(xmesh, ymesh))
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(title)
    plt.savefig(fname)

# for ne, f_list in enumerate(state_lists):
#     edge_plot(f_list , t_list, "edge "+str_edge(ne),"./fano_plane/edge_"+str_edge(ne)+".png")


def build_rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])


def build_stretch_matrix(sx, sy=None):
    if sy is None:
        sy = sx
    return np.array([[sx, 0],
                     [0, sy]])


def build_line_transform(L, R, sy=None):
    """Build affine transform moving unit interval to (L, R) with sy stretch."""
    # Find angle and distance between points
    dx = R[0] - L[0]
    dy = R[1] - L[1]
    theta = np.arctan2(dy, dx)
    sx = np.sqrt(dx**2 + dy**2)
    # Build affine matrices
    stretch = build_stretch_matrix(sx, sy)
    rotation = build_rotation_matrix(theta)
    A = rotation @ stretch
    b = np.array(L)
    return A, b


def apply_line_transform(x, y, L, R, sy=None):
    """Apply affine line transform."""
    A, b = build_line_transform(L, R, sy)
    X = np.array([x, y])
    Y = A@X + b[:,None]
    return Y[0], Y[1]


def graphplot(filename, start, count, output):

    # Parameters
    dpi = 100
    amp_stretch = 0.15
    title = False

    # Make frames
    fig = plt.figure(figsize=(10,10))
    with h5py.File(filename, mode='r') as file:
        x = file['scales']['x']['1.0'][:]
        t = file['scales']['sim_time']
        for index in range(start, start+count):
            axes = fig.add_axes([0.05, 0.05, 0.9, 0.9])
            for ne, edge in enumerate(edges):
                # Load edge solution
                L = verts[edge[0]]
                R = verts[edge[1]]
                u = file['tasks'][str_u(ne)][index]
                y = np.abs(u)
                # Plot symmetric and fill
                xt, yt = apply_line_transform(x/10, y, L, R, amp_stretch)
                xb, yb = apply_line_transform(x/10, -y, L, R, amp_stretch)
                axes.fill(np.concatenate((xt, xb[::-1])), np.concatenate((yt, yb[::-1])), ec='none', fc='k', alpha=0.5)
            axes.set_xlim(-1.1, 1.1)
            axes.set_ylim(-1.1, 1.1)
            axes.axis('off')
            if title:
                fig.suptitle('%.2f' %t[index], fontsize='large')
            # Save frame
            savename = 'graph_%06i.png' %file['scales/write_number'][index]
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
            fig.clear()


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], graphplot, output=output_path)

