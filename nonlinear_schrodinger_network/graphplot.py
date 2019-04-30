"""Tools for plotting graph."""

import numpy as np


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
def plot_graph(file, index, axes, amp_stretch=1, amp_offset=0, lw=0, lc='k', ec='none', fc='k', alpha=0.5):
    x = file['scales']['x']['2'][:]
    for ne, edge in enumerate(edges):
        # Load edge solution
        L = verts[edge[0]]
        R = verts[edge[1]]
        u = file['tasks'][str_u(ne)][index]
        y = np.abs(u) * amp_stretch + amp_offset
        # Plot symmetric and fill
        xt, yt = apply_line_transform(x, y, L, R)
        xb, yb = apply_line_transform(x, -y, L, R)
        if lw:
            axes.plot((L[0], R[0]), (L[1], R[1]), '--', lw=lw, c=lc, dashes=(10, 10))
        axes.fill(np.concatenate((xt, xb[::-1])), np.concatenate((yt, yb[::-1])), ec=ec, fc=fc, alpha=alpha)

