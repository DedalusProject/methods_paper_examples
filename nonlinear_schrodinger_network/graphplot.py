"""Plot output from NLS simulation."""

import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import shelve
import pathlib

# Prevent running from dropbox
path = pathlib.Path(__file__).resolve()
if 'dropbox' in str(path).lower():
    raise RuntimeError("It looks like you're running this script inside a dropbox folder. This has been disallowed to prevent spamming other shared-folder users.")


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

def apply_arc_transform(x, y, L, R, cy, sy=None):
    """Apply affine+arc transform."""
    if cy is None:
        return apply_line_transform(x, y, L, R, sy=sy)
    # Circle center: (1/2, cy)
    # Probably only works for cy < 0
    cx = 1/2
    θlim = np.arctan2(1/2, -cy)
    cr = np.sqrt(cx**2 + cy**2)
    # Apply y stretch
    X = np.array([x, y])
    S = build_stretch_matrix(1, sy)
    SX = S@X
    sx, sy = SX[0], SX[1]
    # Convert to polar coordinates
    sθ = (2*sx - 1) * θlim
    sr = cr + sy
    sx = cx + sr*np.sin(sθ)
    sy = cy + sr*np.cos(sθ)
    # Apply line transform
    return apply_line_transform(sx, sy, L, R)


# Points
r3 = np.sqrt(3)
p = {}
p[0] = [-r3, -1]
p[1] = [-r3/2, 1/2]
p[2] = [0, 2]
p[3] = [r3/2, 1/2]
p[4] = [r3, -1]
p[5] = [0, -1]
p[6] = [0, 0]

# Segments
# beginning point, end point, cy
θmax = 2 * np.pi / 6
cy = - 1 / 2 / np.tan(θmax)
s = {}
s['01'] = [p[0], p[1], None]
s['12'] = [p[1], p[2], None]
s['23'] = [p[2], p[3], None]
s['34'] = [p[3], p[4], None]
s['45'] = [p[4], p[5], None]
s['50'] = [p[0], p[5], None] # uh oh
s['13'] = [p[1], p[3], cy]
s['35'] = [p[3], p[5], cy]
s['51'] = [p[5], p[1], cy]
s['06'] = [p[0], p[6], None]
s['16'] = [p[1], p[6], None]
s['26'] = [p[2], p[6], None]
s['36'] = [p[3], p[6], None]
s['46'] = [p[4], p[6], None]
s['56'] = [p[5], p[6], None]

# Setup output folder
folder = './frames'
if os.path.exists(folder):
    shutil.rmtree(folder)
os.mkdir(folder)

# Load data
filename = 'edges.dat'
data = shelve.open(filename, flag='r', protocol=-1)
x = data['x']
t = data['t']

# Make frames
sy = 0.15
fig = plt.figure(figsize=(10,10))
for index in range(0, t.size):
    print('Plotting index', index)
    axes = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    for seg in s:
        L, R, cy = s[seg]
        y = data['u{}'.format(seg)][index]
        xt, yt = apply_arc_transform(x/10, y, L, R, cy, sy)
        #axes.plot(xt, yt, '-k')
        xb, yb = apply_arc_transform(x/10, -y, L, R, cy, sy)
        #axes.plot(xb, yb, '-k')
        axes.fill(np.concatenate((xt, xb[::-1])), np.concatenate((yt, yb[::-1])), ec='none', fc='k', alpha=0.5)
    axes.set_xlim(-1.8, 1.8)
    axes.set_ylim(-1.3, 2.3)
    axes.axis('off')
    fig.suptitle('%.2f' %t[index], fontsize='large')
    fig.savefig(os.path.join(folder, 'frame_%05i.png' %index), dpi=100)
    fig.clear()
