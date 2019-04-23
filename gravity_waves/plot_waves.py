import matplotlib.pyplot as plt
import numpy as np
import h5py

with h5py.File('wave_frequencies.h5','r') as infile:
    freqs = [k.value for k in infile.values() if 'freq' in k.name]
    N = len(freqs)
    print("Read {}-frequency sets".format(N))
    ks = infile['/grid'][:]
    brunt = infile['brunt'][()]

fig, ax = plt.subplots(nrows=2, ncols=1)
for i, k in enumerate(ks):
    y = freqs[i]
    no_conv_mask = np.where(np.abs(y.real)<1e-4)
    ks = np.array([k]*len(y))
    ω = np.abs(y.imag)
    gwaves = np.where(ω <= brunt)
    acoustic = np.where(ω > brunt)
    #ax.plot(ks, np.abs(y.real), marker='o', linestyle='none')
    ax[0].plot(ks[acoustic], ω[acoustic], marker='o', linestyle='none')
    ax[0].plot(ks[gwaves], ω[gwaves], marker='x', linestyle='none')
    ax[1].plot(ks[acoustic], 1/ω[acoustic], marker='o', linestyle='none')
    ax[1].plot(ks[gwaves], 1/ω[gwaves], marker='x', linestyle='none')
ax[0].axhline(y=brunt, linestyle='dashed', color='black')
ax[1].axhline(y=1/brunt, linestyle='dashed', color='black')
ax[0].set_xlabel('wavenumber k')
ax[0].set_ylabel(r'frequency $\omega$')
ax[0].set_ylim(0,15)
ax[1].set_xlabel('wavenumber k')
ax[1].set_ylabel(r'Period $1/\omega$')
ax[1].set_ylim(1/brunt*0.9,1/brunt*5.1)

fig.savefig('frequency_spectrum.png', dpi=600)
plt.show()
