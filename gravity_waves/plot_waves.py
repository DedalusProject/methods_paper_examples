import matplotlib.pyplot as plt
import numpy as np
import h5py

with h5py.File('wave_frequencies.h5','r') as infile:
    freqs = [k.value for k in infile.values() if 'freq' in k.name]
    N = len(freqs)
    print("Read {}-frequency sets".format(N))
    ks = infile['/grid'][:]

fig, ax = plt.subplots()
for i, k in enumerate(ks):
    y = freqs[i]
    no_conv_mask = np.where(np.abs(y.real)<1e-4)
    #y = y[no_conv_mask]
    ax.plot([k]*len(y), np.abs(y.real), marker='o', linestyle='none')
    ax.plot([k]*len(y), np.abs(y.imag), marker='x', linestyle='none')
#ax.axhline(y=brunt, linestyle='dashed', color='black')
ax.set_xlabel('wavenumber k')
ax.set_ylabel('frequency $\omega$')
#ax.set_yscale('log')
plt.show()
