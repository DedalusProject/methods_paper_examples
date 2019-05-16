import matplotlib.pyplot as plt
import numpy as np
import h5py

with h5py.File('wave_frequencies.h5','r') as infile:
    freqs = []
    eigs_w = []
    eigs_u = []
    eigs_T = []
    ω_plus_min = []
    ω_minus_max = []

    for k_i in infile['tasks']:
        freqs.append(infile['tasks'][k_i]['freq'][:])
        eigs_w.append(infile['tasks'][k_i]['eig_w'][:])
        eigs_u.append(infile['tasks'][k_i]['eig_u'][:])
        eigs_T.append(infile['tasks'][k_i]['eig_T'][:])
        ω_plus_min.append(infile['tasks'][k_i]['ω_plus_min'][()])
        ω_minus_max.append(infile['tasks'][k_i]['ω_minus_max'][()])

    ks = infile['scales']['grid'][:]
    brunt = infile['scales']['brunt_max'][()]
    k_Hρ  = infile['scales']['k_Hρ'][()]
    c_s   = infile['scales']['c_s'][()]
    z = infile['scales']['z'][:]
    rho0 = infile['scales']['rho0'][:]
    Lz = infile['scales']['Lz'][()]

    N = len(freqs)
    print("Read {}-frequency sets".format(N))

ω_upper = np.array(ω_plus_min)/brunt
ω_lower = np.array(ω_minus_max)/brunt

#ks /= k_Hρ
target_k = 17.5
i_k = np.argmin(np.abs(ks-target_k))
print(ω_upper)
print(ω_lower)
pad = 1.05

c_acoustic = 'lightskyblue'
c_gravity = 'firebrick'
c_f_mode = 'darkslategrey'
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

for i, k1 in enumerate(ks):
    ω = freqs[i].real/brunt
    i_sort = np.argsort(ω)
    ω = ω[i_sort]
    P = 1/ω

    k = np.array([k1]*len(ω))

    gwaves = np.where(np.abs(ω) <= ω_lower[i]*pad)
    f_mode = np.where(np.logical_and(np.abs(ω) <= ω_upper[i], np.abs(ω) > ω_lower[i])*pad)
    acoustic = np.where(np.abs(ω) > ω_upper[i])

    ax[0].plot(k[acoustic], np.abs(ω[acoustic]), marker='x', linestyle='none', color=c_acoustic, zorder=2)
    ax[0].plot(k[f_mode], np.abs(ω[f_mode]), marker='x', linestyle='none', color=c_f_mode, zorder=2)
    ax[0].plot(k[gwaves], np.abs(ω[gwaves]), marker='x', linestyle='none', color=c_gravity, zorder=2)
    if i == 0:
        ax[1].plot(k[gwaves], np.abs(1/ω[gwaves]), marker='x', linestyle='none', color=c_gravity, zorder=2, label='gw')
        ax[1].plot(k[f_mode], np.abs(1/ω[f_mode]), marker='x', linestyle='none', color=c_f_mode, zorder=2, label='f')
        ax[1].plot(k[acoustic], np.abs(1/ω[acoustic]), marker='x', linestyle='none', color=c_acoustic, zorder=2, label='ac')
    else:
        ax[1].plot(k[acoustic], np.abs(1/ω[acoustic]), marker='x', linestyle='none', color=c_acoustic, zorder=2)
        ax[1].plot(k[f_mode], np.abs(1/ω[f_mode]), marker='x', linestyle='none', color=c_f_mode, zorder=2)
        ax[1].plot(k[gwaves], np.abs(1/ω[gwaves]), marker='x', linestyle='none', color=c_gravity, zorder=2)
    if i == i_k:
        print('Eigenfunctions for kx = {:g} ({:g} '.format(k1*k_Hρ, k1)+r'$k_{H\rho}$)')
        fig_eig, ax_eig = plt.subplots(nrows=2, sharex=True)
        for axR in ax_eig:
            axR.set_ylabel(r'$\sqrt{\rho}w,\,\sqrt{\rho}u, \sqrt{\rho}T$')

        i_brunt = np.argmin(np.abs(P-1/ω_lower[i]*pad))
        w = eigs_w[i][i_sort,:]
        u = eigs_u[i][i_sort,:]
        T = eigs_T[i][i_sort,:]
        gw = -4
        ac = 5
        weight = np.sqrt(rho0)
        wc = 'steelblue'
        ax_eig[1].plot(z, weight*w[i_brunt+gw,:].real, color=wc)
        ax_eig[0].plot(z, weight*w[i_brunt+ac,:].real, color=wc)
        ax_eig[1].plot(z, weight*w[i_brunt+gw,:].imag, linestyle='dashed', color=wc)
        ax_eig[0].plot(z, weight*w[i_brunt+ac,:].imag, linestyle='dashed', color=wc)

        uc = 'seagreen'
        ax_eig[1].plot(z, weight*u[i_brunt+gw,:].real, color=uc)
        ax_eig[0].plot(z, weight*u[i_brunt+ac,:].real, color=uc)
        ax_eig[1].plot(z, weight*u[i_brunt+gw,:].imag, linestyle='dashed', color=uc)
        ax_eig[0].plot(z, weight*u[i_brunt+ac,:].imag, linestyle='dashed', color=uc)

        Tc = 'firebrick'
        ax_eig[1].plot(z, weight*T[i_brunt+gw,:].real, color=Tc)
        ax_eig[0].plot(z, weight*T[i_brunt+ac,:].real, color=Tc)
        ax_eig[1].plot(z, weight*T[i_brunt+gw,:].imag, linestyle='dashed', color=Tc)
        ax_eig[0].plot(z, weight*T[i_brunt+ac,:].imag, linestyle='dashed', color=Tc)

        ax_eig[1].text(0, 1, 'gravity waves\n'+r'($\omega \leq \omega_-$)', verticalalignment='top', multialignment='center')
        ax_eig[0].text(0, 1, 'acoustic waves\n'+r'($\omega > \omega_+$)', verticalalignment='top', multialignment='center')

        print("gw {:g}, {:g}".format(ω[i_brunt+gw], 1/ω[i_brunt+gw]))
        print("ac {:g}, {:g}".format(ω[i_brunt+ac], 1/ω[i_brunt+ac]))

        fig_eig, ax_eig = plt.subplots(nrows=2, sharex=True)
        for axR in ax_eig:
            axR.set_ylabel(r'$\sqrt{\rho}w$')
            axR.axhline(y=0, color='black', linestyle='dashed')

        gws = [0, -2, -4, -8]
        acs = [1, 3, 5, 9]
        weight = np.sqrt(rho0)

        colors = ['firebrick', 'darkorange', 'seagreen', 'darkslateblue']
        i_color = 0
        print("   ω/N, N/ω")
        for gw, ac in zip(gws,acs):
            ax_eig[0].plot(z, weight*w[i_brunt+ac,:].real, color=colors[i_color], label='{:3.1f}'.format(ω[i_brunt+ac]))
            ax_eig[1].plot(z, weight*w[i_brunt+gw,:].real, color=colors[i_color], label='{:3.1f}'.format(1/ω[i_brunt+gw]))

            ax[0].plot(k[0], ω[i_brunt+gw], marker='o', color=colors[i_color], alpha=0.5, markersize=10, zorder=3)
            ax[0].plot(k[0], ω[i_brunt+ac], marker='o', color=colors[i_color], alpha=0.5, markersize=10, zorder=3)
            ax[1].plot(k[0], P[i_brunt+gw], marker='o', color=colors[i_color], alpha=0.5, markersize=10, zorder=3)
            ax[1].plot(k[0], P[i_brunt+ac], marker='o', color=colors[i_color], alpha=0.5, markersize=10, zorder=3)
            print("gw {:3.1f}, {:3.1f}".format(ω[i_brunt+gw], 1/ω[i_brunt+gw]))
            print("ac {:3.1f}, {:3.1f}".format(ω[i_brunt+ac], 1/ω[i_brunt+ac]))
            i_color += 1

        legend = ax_eig[1].legend(frameon=False, title=r'period $N/\omega$', loc='lower left', ncol=4)
        for line,text in zip(legend.get_lines(), legend.get_texts()):
            text.set_color(line.get_color())
        legend = ax_eig[0].legend(frameon=False, title=r'frequency $\omega/N$', loc='lower left', ncol=4)
        for line,text in zip(legend.get_lines(), legend.get_texts()):
            text.set_color(line.get_color())
        ax_eig[1].text(0, 1, 'gravity waves\n'+r'($\omega \leq \omega_-$)', verticalalignment='top', multialignment='center')
        ax_eig[0].text(0, 1, 'acoustic waves\n'+r'($\omega > \omega_+$)', verticalalignment='top', multialignment='center')
        ax_eig[1].set_xlabel(r'height $z$')

ax[0].plot(ks, ω_lower, color=c_gravity, linestyle='dashed', zorder=1)
ax[0].plot(ks, ω_upper, color=c_acoustic, linestyle='dashed', zorder=1)
ax[0].axhline(y=1, linestyle='dashed', color='black', zorder=1)
ax[0].set_ylabel(r'frequency $\omega/N$')
ax[0].set_ylim(0,5)
ax[0].set_xscale('log')

ax[1].plot(ks, 1/ω_lower, color=c_gravity, linestyle='dashed', label=r'$\omega_-$', zorder=1)
ax[1].axhline(y=1, linestyle='dashed', color='black', label=r'$N$', zorder=1)
ax[1].plot(ks, 1/ω_upper, color=c_acoustic, linestyle='dashed', label=r'$\omega_+$', zorder=1)
ax[1].set_xlabel(r'wavenumber $k$')
ax[1].set_ylabel(r'period $N/\omega$')
ax[1].set_ylim(0,5.1)
ax[1].set_xscale('log')
legend = ax[1].legend(ncol=2, framealpha=0.7)
legend.get_frame().set_linewidth(0.0)

plt.tight_layout()
plt.subplots_adjust(hspace=0.05)

fig.savefig('wave_frequency_spectrum.pdf')
fig_eig.savefig('wave_eigenfunctions.pdf')
plt.show()
