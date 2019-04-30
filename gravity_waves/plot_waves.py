import matplotlib.pyplot as plt
import numpy as np
import h5py

with h5py.File('wave_frequencies.h5','r') as infile:
    freqs = [k.value for k in infile.values() if 'freq' in k.name]
    N = len(freqs)
    print("Read {}-frequency sets".format(N))
    ks = infile['/grid'][:]
    brunt = infile['brunt'][()]
    k_Hρ  = infile['k_Hrho'][()]
    c_s   = infile['c_s'][()]

Lz = 5
kz = 2*np.pi/Lz #np.pi/Lz
ω2_sw = (ks**2 + kz**2 + k_Hρ**2)*c_s**2
ω2_gw = ks**2/(ks**2 + kz**2 + k_Hρ**2)*brunt**2
ω_upper = np.sqrt(ω2_sw/2*(1+np.sqrt(1-4*ω2_gw/ω2_sw)))/brunt
ω_lower = np.sqrt(ω2_sw/2*(1-np.sqrt(1-4*ω2_gw/ω2_sw)))/brunt
print(ω2_sw/brunt**2)
print(ω2_gw/brunt**2)
print(ω_upper)
print(ω_lower)

print(ks)
ks /= k_Hρ
print(ks)
#freqs /= brunt

c_acoustic = 'lightskyblue'
c_gravity = 'firebrick'
fig, ax = plt.subplots(nrows=2, ncols=1)
for i, k1 in enumerate(ks):
    y = freqs[i]
    k = np.array([k1]*len(y))
    oscillatory = np.where(np.abs(y.imag)<1e-2)
    ω = np.abs(y[oscillatory].real)/brunt
    gwaves = np.where(ω <= 1)
    acoustic = np.where(ω > 1)
    #ax.plot(ks, np.abs(y.real), marker='o', linestyle='none')
    ax[0].plot(k[acoustic], ω[acoustic], marker='x', linestyle='none', color=c_acoustic)
    ax[0].plot(k[gwaves], ω[gwaves], marker='x', linestyle='none', color=c_gravity)
    ax[1].plot(k[acoustic], 1/ω[acoustic], marker='x', linestyle='none', color=c_acoustic)
    ax[1].plot(k[gwaves], 1/ω[gwaves], marker='x', linestyle='none', color=c_gravity)
ax[0].plot(ks, ω_upper, linestyle='dashed')
ax[0].plot(ks, ω_lower, linestyle='dashed')
ax[0].axhline(y=1, linestyle='dashed', color='black')
ax[1].axhline(y=1, linestyle='dashed', color='black')
ax[0].set_xlabel(r'wavenumber $k/k_{H\rho}$')
ax[0].set_ylabel(r'frequency $\omega/N$')
ax[0].set_ylim(0,5)
#ax[0].set_yscale('log')
#ax[0].set_ylim(0,15)
ax[1].set_xlabel(r'wavenumber $k/k_{H\rho}$')
ax[1].set_ylabel(r'Period $N/\omega$')
ax[1].set_ylim(0,5.1)
ax[0].set_xscale('log')
ax[1].set_xscale('log')

fig.savefig('frequency_spectrum.png', dpi=600)
plt.show()
