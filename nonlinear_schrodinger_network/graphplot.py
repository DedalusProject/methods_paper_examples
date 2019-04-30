"""
Plot output from NLS simulation.

Usage:
    graphplot.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
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

# Helper functions
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

# Plotting
def plot_graph(file, index, axes, amp_stretch):
    x = file['scales']['x']['1.0'][:]
    for ne, edge in enumerate(edges):
        # Load edge solution
        L = verts[edge[0]]
        R = verts[edge[1]]
        u = file['tasks'][str_u(ne)][index]
        y = np.abs(u) * amp_stretch
        # Plot symmetric and fill
        xt, yt = apply_line_transform(x, y, L, R)
        xb, yb = apply_line_transform(x, -y, L, R)
        axes.fill(np.concatenate((xt, xb[::-1])), np.concatenate((yt, yb[::-1])), ec='none', fc='k', alpha=0.5)

def plot_writes(filename, start, count, output, axes=None, save=True, dpi=100, amp_stretch=0.01, title=False):
    # Make axes if not provided
    if not axes:
        fig = plt.figure(figsize=(10,10))
        axes = fig.add_axes([0, 0, 1, 1])
    # Loop over assigned writes
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            plot_graph(file, index, axes, amp_stretch)
            # Remove axes
            axes.set_xlim(-1.2, 1.2)
            axes.set_ylim(-1.2, 1.2)
            axes.axis('off')
            # Timestamp title
            if title:
                axes.set_title('%.2f' %file['scales']['sim_time'][index], fontsize='large')
            # Save frame
            if save:
                savename = 'graph_%06i.png' %file['scales/write_number'][index]
                savepath = output.joinpath(savename)
                fig.savefig(str(savepath), dpi=dpi)
            axes.cla()


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
    post.visit_writes(args['<files>'], plot_writes, output=output_path)

