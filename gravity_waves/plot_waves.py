import matplotlib.pyplot as plt
import numpy as np
import h5py

with h5py.File('wave_frequencies.h5','r') as infile:
    freqs = []
    eigs_w = []
    eigs_u = []
    eigs_T = []
    for k_i in infile['tasks']:
        freqs.append(infile['tasks'][k_i]['freq'][:])
        eigs_w.append(infile['tasks'][k_i]['eig_w'][:])
        eigs_u.append(infile['tasks'][k_i]['eig_u'][:])
        eigs_T.append(infile['tasks'][k_i]['eig_T'][:])
    ks = infile['scales']['grid'][:]
    brunt = infile['scales']['brunt'][()]
    k_Hρ  = infile['scales']['k_Hρ'][()]
    c_s   = infile['scales']['c_s'][()]
    z = infile['scales']['z'][:]
    rho0 = infile['scales']['rho0'][:]
    Lz = infile['scales']['Lz'][()]

    N = len(freqs)
    print("Read {}-frequency sets".format(N))

Lz_atm = Lz
Lz = Lz_atm
kz = 2*np.pi/Lz #np.pi/Lz
ω2_sw = (ks**2 + kz**2 + k_Hρ**2)*c_s**2
ω2_gw = ks**2/(ks**2 + kz**2 + k_Hρ**2)*brunt**2
ω_upper = np.sqrt(ω2_sw/2*(1+np.sqrt(1-4*ω2_gw/ω2_sw)))/brunt
ω_lower = np.sqrt(ω2_sw/2*(1-np.sqrt(1-4*ω2_gw/ω2_sw)))/brunt

kz = 2*np.pi/Lz_atm #np.pi/Lz
ω2_sw = (ks**2 + kz**2 + k_Hρ**2)*c_s**2
ω2_gw = ks**2/(ks**2 + kz**2 + k_Hρ**2)*brunt**2
ω_upper = np.sqrt(ω2_sw/2*(1+np.sqrt(1-4*ω2_gw/ω2_sw)))/brunt


ks /= k_Hρ

c_acoustic = 'lightskyblue'
c_gravity = 'firebrick'
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
for i, k1 in enumerate(ks):
    y = freqs[i]
    k = np.array([k1]*len(y))

    ω = np.abs(y.real)/brunt
    gwaves = np.where(ω <= ω_lower[i])
    acoustic = np.where(ω > ω_lower[i])

    ax[0].plot(k[acoustic], ω[acoustic], marker='x', linestyle='none', color=c_acoustic)
    ax[0].plot(k[gwaves], ω[gwaves], marker='x', linestyle='none', color=c_gravity)
    ax[1].plot(k[acoustic], 1/ω[acoustic], marker='x', linestyle='none', color=c_acoustic)
    ax[1].plot(k[gwaves], 1/ω[gwaves], marker='x', linestyle='none', color=c_gravity)
    target_k = 23
    i_k = np.argmin(np.abs(ks-target_k))
    if i == i_k:
        fig_eig, ax_eig = plt.subplots(nrows=2, sharex=True)
        for axR in ax_eig:
            axR.set_ylabel(r'$\sqrt{\rho}w,\,\sqrt{\rho}u$')
        ax_eig_L = []
        for axR in ax_eig:
            ax_eig_L.append(axR.twinx())
        for axL in ax_eig_L:
            axL.set_ylabel(r'$\rho T$')


        i_sort = np.argsort(ω)
        P = 1/ω[i_sort]
        i_brunt = np.argmin(np.abs(P-1/ω_lower[i]))
        w = eigs_w[i][i_sort,:]
        u = eigs_u[i][i_sort,:]
        T = eigs_T[i][i_sort,:]
        gw = -19
        ac = 20
        weight = np.sqrt(rho0)
        wc = 'steelblue'
        #ax_eig[0].plot(z, weight*w[i_brunt+gw,:].real, color=wc)
        ax_eig[1].plot(z, weight*w[i_brunt+ac,:].real, color=wc)
        ax_eig[0].plot(z, weight*w[i_brunt+gw,:].imag, linestyle='dashed', color=wc)
        #ax_eig[1].plot(z, weight*w[i_brunt+ac,:].imag, linestyle='dashed', color=wc)

        uc = 'seagreen'
        ax_eig[0].plot(z, weight*u[i_brunt+gw,:].real, color=uc)
        #ax_eig[1].plot(z, weight*u[i_brunt+ac,:].real, color=uc)
        #ax_eig[0].plot(z, weight*u[i_brunt+gw,:].imag, linestyle='dashed', color=uc)
        ax_eig[1].plot(z, weight*u[i_brunt+ac,:].imag, linestyle='dashed', color=uc)

        #weight = rho0
        Tc = 'firebrick'
        ax_eig_L[0].plot(z, weight*T[i_brunt+gw,:].real, color=Tc)
        #ax_eig_L[1].plot(z, weight*T[i_brunt+ac,:].real, color=Tc)
        #ax_eig_L[0].plot(z, weight*T[i_brunt+gw,:].imag, linestyle='dashed', color=Tc)
        ax_eig_L[1].plot(z, weight*T[i_brunt+ac,:].imag, linestyle='dashed', color=Tc)

        #ax[0].plot(k[0], ω[i_sort][i_brunt+gw], marker='o', color='black', alpha=0.2, markersize=10)
        #ax[0].plot(k[0], ω[i_sort][i_brunt+ac], marker='o', color='black', alpha=0.2, markersize=10)
        #ax[1].plot(k[0], P[i_brunt+gw], marker='o', color='black', alpha=0.2, markersize=10)
        #ax[1].plot(k[0], P[i_brunt+ac], marker='o', color='black', alpha=0.2, markersize=10)
        print(ω[i_sort][i_brunt+gw], 1/ω[i_sort][i_brunt+gw])
        print(ω[i_sort][i_brunt+ac], 1/ω[i_sort][i_brunt+ac])

        fig_eig, ax_eig = plt.subplots(nrows=2, sharex=True)
        for axR in ax_eig:
            axR.set_ylabel(r'$\sqrt{\rho}w$')

        gws = [-1, -6, -11, -16]
        acs = [1, 6, 11, 16]
        weight = np.sqrt(rho0)

        colors = ['steelblue', 'seagreen', 'firebrick', 'darkslateblue']
        i_color = 0
        #ax_eig[0].plot(z, weight*w[i_brunt+gw,:].real, color=wc)
        for gw, ac in zip(gws,acs):
            ax_eig[1].plot(z, weight*w[i_brunt+ac,:].real, color=colors[i_color])
            ax_eig[1].plot(z, weight*w[i_brunt+ac,:].imag, linestyle='dashed', color=colors[i_color])
            ax_eig[0].plot(z, weight*w[i_brunt+gw,:].real, color=colors[i_color])
            ax_eig[0].plot(z, weight*w[i_brunt+gw,:].imag, linestyle='dashed', color=colors[i_color])
            i_color += 1

            ax[0].plot(k[0], ω[i_sort][i_brunt+gw], marker='o', color='black', alpha=0.2, markersize=10)
            ax[0].plot(k[0], ω[i_sort][i_brunt+ac], marker='o', color='black', alpha=0.2, markersize=10)
            ax[1].plot(k[0], P[i_brunt+gw], marker='o', color='black', alpha=0.2, markersize=10)
            ax[1].plot(k[0], P[i_brunt+ac], marker='o', color='black', alpha=0.2, markersize=10)
            print(ω[i_sort][i_brunt+gw], 1/ω[i_sort][i_brunt+gw])
            print(ω[i_sort][i_brunt+ac], 1/ω[i_sort][i_brunt+ac])
        ax_eig[0].text(0, 1, 'gravity waves\n'+r'($\omega \leq \omega_\mathrm{GW}$)', verticalalignment='top')
        ax_eig[1].text(0, 1, 'acoustic waves\n'+r'($\omega > \omega_\mathrm{GW}$)', verticalalignment='top')
        ax_eig[1].set_xlabel(r'height $z$')

ax[0].plot(ks, ω_lower, linestyle='dashed')
ax[0].axhline(y=1, linestyle='dashed', color='black')
ax[0].set_ylabel(r'frequency $\omega/N$')
ax[0].set_ylim(0,5)
ax[0].set_xscale('log')

ax[1].plot(ks, 1/ω_lower, linestyle='dashed')
ax[1].axhline(y=1, linestyle='dashed', color='black')
ax[1].set_xlabel(r'wavenumber $k/k_{H\rho}$')
ax[1].set_ylabel(r'Period $N/\omega$')
ax[1].set_ylim(0,5.1)
ax[1].set_xscale('log')

plt.tight_layout()
plt.subplots_adjust(hspace=0.05)

fig.savefig('wave_frequency_spectrum.pdf')
fig_eig.savefig('wave_eigenfunctions.pdf')
plt.show()
