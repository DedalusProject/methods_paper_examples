import matplotlib.pyplot as plt
import numpy as np
import h5py

filename = lambda s: 'snapshots/snapshots_s{:d}.h5'.format(s)

x = np.array([])
y = np.array([])
t = np.array([])
for i in range(4):
    with h5py.File(filename(i+1), 'r') as file:
        x = np.concatenate((x, file['tasks/x'][:,0,0]))
        y = np.concatenate((y, file['tasks/y'][:,0,0]))
        t = np.concatenate((y, file['scales/sim_time'][:]))

fig, ax = plt.subplots()
ax.plot(x,y)

fig.savefig('xy_locations.png', dpi=600)
