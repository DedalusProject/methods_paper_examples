import matplotlib.pyplot as plt
import numpy as np
import h5py

with h5py.File('wave_frequencies.h5','r') as infile:
    freqs = []
    eigs_w = []
    eigs_u = []
    for k_i in infile['tasks']:
        freqs.append(infile['tasks'][k_i]['freq'][:])
        eigs_w.append(infile['tasks'][k_i]['eig_w'][:])
        eigs_u.append(infile['tasks'][k_i]['eig_u'][:])
    ks = infile['scales']['grid'][:]
    brunt = infile['scales']['brunt'][()]
    k_Hρ  = infile['scales']['k_Hρ'][()]
    c_s   = infile['scales']['c_s'][()]
    z = infile['scales']['z'][:]
    rho0 = infile['scales']['rho0'][:]
    Lz = infile['scales']['Lz'][()]

    N = len(freqs)
    print("Read {}-frequency sets".format(N))

#Lz = 1.5 - 0.91 #1.5
kz = 2*np.pi/Lz #np.pi/Lz
ω2_sw = (ks**2 + kz**2 + k_Hρ**2)*c_s**2
ω2_gw = ks**2/(ks**2 + kz**2 + k_Hρ**2)*brunt**2
ω_upper = np.sqrt(ω2_sw/2*(1+np.sqrt(1-4*ω2_gw/ω2_sw)))/brunt
ω_lower = np.sqrt(ω2_sw/2*(1-np.sqrt(1-4*ω2_gw/ω2_sw)))/brunt

ks /= k_Hρ

c_acoustic = 'lightskyblue'
c_gravity = 'firebrick'
fig, ax = plt.subplots(nrows=2, ncols=1)
for i, k1 in enumerate(ks):
    y = freqs[i]
    k = np.array([k1]*len(y))
    oscillatory = np.where(np.abs(y.imag)<1e-8)
    ω = np.abs(y[oscillatory].real)/brunt
    gwaves = np.where(ω <= ω_lower[i])
    acoustic = np.where(ω > ω_lower[i])
    #ax.plot(ks, np.abs(y.real), marker='o', linestyle='none')
    ax[0].plot(k[acoustic], ω[acoustic], marker='x', linestyle='none', color=c_acoustic)
    ax[0].plot(k[gwaves], ω[gwaves], marker='x', linestyle='none', color=c_gravity)
    ax[1].plot(k[acoustic], 1/ω[acoustic], marker='x', linestyle='none', color=c_acoustic)
    ax[1].plot(k[gwaves], 1/ω[gwaves], marker='x', linestyle='none', color=c_gravity)
    target_k = 2
    i_k = np.argmin(np.abs(ks-target_k))
    if i == i_k:
         fig_eig, ax_eig = plt.subplots(nrows=3)
         i_sort = np.argsort(ω)
         P = 1/ω[i_sort]
         i_brunt = np.argmin(np.abs(P-1/ω_lower[i]))
         w = eigs_w[i][i_sort,:]
         u = eigs_u[i][i_sort,:]
         gw = -5
         ac = 5
         mix = 1
         weight = np.sqrt(rho0)
         ax_eig[0].plot(z, weight*w[i_brunt+gw,:].real)
         ax_eig[1].plot(z, weight*w[i_brunt+mix,:].real)
         ax_eig[2].plot(z, weight*w[i_brunt+ac,:].real)
         ax_eig[0].plot(z, weight*w[i_brunt+gw,:].imag)
         ax_eig[1].plot(z, weight*w[i_brunt+mix,:].imag)
         ax_eig[2].plot(z, weight*w[i_brunt+ac,:].imag)
         for axR in ax_eig:
             axR.set_ylabel(r'$\sqrt{\rho}w$')
         ax_eig_L = []
         for axR in ax_eig:
            ax_eig_L.append(axR.twinx())
         ax_eig_L[0].plot(z, weight*u[i_brunt+gw,:].real, linestyle='dashed')
         ax_eig_L[1].plot(z, weight*u[i_brunt+mix,:].real, linestyle='dashed')
         ax_eig_L[2].plot(z, weight*u[i_brunt+ac,:].real, linestyle='dashed')
         ax_eig_L[0].plot(z, weight*u[i_brunt+gw,:].imag, linestyle='dashed')
         ax_eig_L[1].plot(z, weight*u[i_brunt+mix,:].imag, linestyle='dashed')
         ax_eig_L[2].plot(z, weight*u[i_brunt+ac,:].imag, linestyle='dashed')
         for axL in ax_eig_L:
             axL.set_ylabel(r'$\sqrt{\rho}u$')
         ax[1].plot(k[0], P[i_brunt+gw], marker='o', color='black', alpha=0.2, markersize=10)
         ax[1].plot(k[0], P[i_brunt+mix], marker='o', color='black', alpha=0.2, markersize=10)
         ax[1].plot(k[0], P[i_brunt+ac], marker='o', color='black', alpha=0.2, markersize=10)

ax[0].plot(ks, ω_upper, linestyle='dashed')
ax[0].plot(ks, ω_lower, linestyle='dashed')
ax[0].axhline(y=1, linestyle='dashed', color='black')
ax[1].axhline(y=1, linestyle='dashed', color='black')
ax[0].set_xlabel(r'wavenumber $k/k_{H\rho}$')
ax[0].set_ylabel(r'frequency $\omega/N$')
ax[0].set_ylim(0,5)
#ax[0].set_yscale('log')
#ax[0].set_ylim(0,15)
ax[1].plot(ks, 1/ω_lower, linestyle='dashed')
ax[1].set_xlabel(r'wavenumber $k/k_{H\rho}$')
ax[1].set_ylabel(r'Period $N/\omega$')
ax[1].set_ylim(0,5.1)
ax[0].set_xscale('log')
ax[1].set_xscale('log')

fig.savefig('frequency_spectrum.png', dpi=600)
plt.show()
