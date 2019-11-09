
import numpy as np
import h5py
from mayavi import mlab
from dedalus.extras import plot_tools
from pyface.api import GUI

mlab.options.offscreen = True

# Load data
with h5py.File('slices/slices_s40.h5', 'r') as file:
    x = file['scales']['x']['4'][:]
    y = file['scales']['y']['4'][:]
    z = file['scales']['z']['4'][:]
    b_z = file['tasks']['buoyancy-top'][0,:,:,0]
    b_x = file['tasks']['buoyancy-xslice'][0,0,:,:]
    b_y = file['tasks']['buoyancy-yslice'][0,:,0,:]
    pv_z = file['tasks']['PV-top'][0,:,:,0]
    pv_x = file['tasks']['PV-xslice'][0,0,:,:]
    pv_y = file['tasks']['PV-yslice'][0,:,0,:]

# Vertices
vx = plot_tools.get_1d_vertices(x)
vy = plot_tools.get_1d_vertices(y)
vz = plot_tools.get_1d_vertices(z)

def boxplot(vx, vy, vz, fx, fy, fz, cmap, vmin, vmax):
    # z plane
    X, Y = np.meshgrid(vx, vy, indexing='ij')
    Z = 0*X + vz[-1]
    S = 0*X
    S[:-1, :-1] = fz
    m1 = mlab.mesh(X, Y, Z, scalars=S, colormap=cmap, vmin=vmin, vmax=vmax)
    # x plane
    Y, Z = np.meshgrid(vy, vz, indexing='ij')
    X = 0*Y + vx[0]
    S = 0*X
    S[:-1, :-1] = fx
    m2 = mlab.mesh(X, Y, Z, scalars=S, colormap=cmap, vmin=vmin, vmax=vmax)
    # x plane
    X, Z = np.meshgrid(vx, vz, indexing='ij')
    Y = 0*Z + vy[0]
    S = 0*X
    S[:-1, :-1] = fy
    m3 = mlab.mesh(X, Y, Z, scalars=S, colormap=cmap, vmin=vmin, vmax=vmax)
    return (m1, m2, m3)

# Plot
fig = mlab.figure(bgcolor=(1,1,1), size=(1600, 800))
m1, m2, m3 = boxplot(vx, vy, 2*(vz-3.5), -b_x, -b_y, -b_z, 'RdBu', -20, 20)
m1, m2, m3 = boxplot(vx, vy, 2*(vz+3.5), pv_x, pv_y, pv_z, 'viridis', -60, 60)
mlab.view(azimuth=-140, elevation=70, distance=70, focalpoint=(20,12,-2))
GUI().process_events()
for light in m1.scene.light_manager.lights:
    light.move_to(elevation=20, azimuth=0)
    light.intensity = 0.5
GUI().process_events()
mlab.savefig('fig_QG3D.jpg', magnification=1)

