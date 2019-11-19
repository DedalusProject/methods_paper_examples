"""
Plot waves

Produces eigenvalue vs horizontal wavenumber plots of waves for all computed horizontal and vertical wavenumbers.  This is analagous to a power spectrum, but without amplitude information.  Both a frequency (ω) and period (1/ω) diagram are produced, as acoustic and internal gravity waves have different patterns in those two diagrams.  Modes are automatically classified into "acoustic" or "gravity" wave branches based on asymptotic ω+ and ω- relationships from Hindman & Zweibel 1994 ApJ and colored appropriately.  An f-mode is also identified.  The resulting diagrams, as used in the methods paper, are stored in "wave_frequency_spectrum.pdf".  Additionally, vertical velocity eigenfunction figures are created for acoustic and gravity wave modes, with coloring matching selected modes in the frequency/period diagrams.  These figures, used in the methods paper, are stored in "wave_eigenfunctions.pdf".

It should take approximately 3 seconds on 1 Skylake core.
"""
import matplotlib.pyplot as plt
import numpy as np
import h5py
plt.style.use('./methods_paper.mplstyle')


# Load data
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

# Normalize frequencies
ω_upper = np.array(ω_plus_min) / brunt
ω_lower = np.array(ω_minus_max) / brunt

# Pick kx for mode plotting
target_k = 17.5
i_k = np.argmin(np.abs(ks - target_k))
pad = 1.05

# Frequency parameters
c_acoustic = 'C0'
c_gravity = 'C3'
c_f_mode = 'k'
marker = 'x'
ms = 3
mew = 0.5
freq_props = dict(marker=marker, ms=ms, mew=mew, linestyle='none', zorder=2)

# Figures
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(3.4, 2.8))
fig_eig, ax_eig = plt.subplots(nrows=2, sharex=True, figsize=(3.4, 2.8))
fig.subplots_adjust(left=0.12, bottom=0.12, right=0.97, top=0.97, hspace=0.06)
fig_eig.subplots_adjust(left=0.12, bottom=0.12, right=0.97, top=0.97, hspace=0.06)

# Loop over kx
for i, k1 in enumerate(ks):

    # Compute frequencies
    ω = freqs[i].real/brunt
    i_sort = np.argsort(ω)
    ω = ω[i_sort]
    P = 1 / ω
    k = np.array([k1]*len(ω))

    gwaves = (np.abs(ω) <= ω_lower[i]*pad) * (np.abs(ω) > 0)
    f_mode = (np.abs(ω) <= ω_upper[i]) * (np.abs(ω) > ω_lower[i]*pad)
    acoustic = (np.abs(ω) > ω_upper[i])

    # Plot frequencies
    if i == 0:
        ax[0].plot(k[acoustic], np.abs(ω[acoustic]), color=c_acoustic, label='ac', **freq_props)
        ax[0].plot(k[f_mode], np.abs(ω[f_mode]), color=c_f_mode, label='f', **freq_props)
        ax[0].plot(k[gwaves], np.abs(ω[gwaves]), color=c_gravity, label='ac', **freq_props)
        ax[1].plot(k[gwaves], np.abs(1/ω[gwaves]), color=c_gravity, label='gw', **freq_props)
        ax[1].plot(k[f_mode], np.abs(1/ω[f_mode]), color=c_f_mode, label='f', **freq_props)
        ax[1].plot(k[acoustic], np.abs(1/ω[acoustic]), color=c_acoustic, label='ac', **freq_props)
    else:
        ax[0].plot(k[acoustic], np.abs(ω[acoustic]), color=c_acoustic, **freq_props)
        ax[0].plot(k[f_mode], np.abs(ω[f_mode]), color=c_f_mode, **freq_props)
        ax[0].plot(k[gwaves], np.abs(ω[gwaves]), color=c_gravity, **freq_props)
        ax[1].plot(k[acoustic], np.abs(1/ω[acoustic]), color=c_acoustic,**freq_props)
        ax[1].plot(k[f_mode], np.abs(1/ω[f_mode]), color=c_f_mode, **freq_props)
        ax[1].plot(k[gwaves], np.abs(1/ω[gwaves]), color=c_gravity, **freq_props)

    # Plot eigenfunctions
    if i == i_k:
        print('Eigenfunctions for kx = {:g} ({:g} '.format(k1*k_Hρ, k1)+r'$k_{H\rho}$)')

        # Pick modes
        i_brunt = np.argmin(np.abs(P-1/ω_lower[i]*pad))
        w = eigs_w[i][i_sort,:]
        gws = [0, -2, -4, -8]
        acs = [1, 3, 5, 9]
        weight = np.sqrt(rho0)

        # Mode properties
        colors = ['C1', 'C2', 'C4', 'C5']
        marker = 'o'
        msc = 6
        mec = "none"
        print("   ω/N, N/ω")
        mode_props = dict(marker=marker, mec=mec, markersize=msc, alpha=0.5, zorder=3)

        for i, (gw, ac) in enumerate(zip(gws,acs)):
            igw = i_brunt + gw
            iac = i_brunt + ac
            print("gw {:3.1f}, {:3.1f}".format(ω[igw], 1/ω[igw]))
            print("ac {:3.1f}, {:3.1f}".format(ω[iac], 1/ω[iac]))
            # Plot modes
            ax_eig[0].plot(z, weight*w[iac,:].real, color=colors[i], label='{:3.1f}'.format(ω[iac]))
            ax_eig[1].plot(z, weight*w[igw,:].real, color=colors[i], label='{:3.1f}'.format(1/ω[igw]))
            # Highlight frequencies
            mode_props['color'] = colors[i]
            ax[0].plot(k[0], ω[igw], **mode_props)
            ax[0].plot(k[0], ω[iac], **mode_props)
            ax[1].plot(k[0], P[igw], **mode_props)
            ax[1].plot(k[0], P[iac], **mode_props)

# Frequency lines
line_props = {'ls': 'solid', 'lw': 1, 'alpha': 0.5, 'zorder': 1}
ax[0].plot(ks, ω_lower, color=c_gravity, label=r'$\omega_-$', **line_props)
ax[0].plot(ks, ω_upper, color=c_acoustic, label=r'$\omega_+$', **line_props)
#ax[0].axhline(y=1, color='black', **line_props)
ax[1].plot(ks, 1/ω_lower, color=c_gravity, label=r'$\omega_-$', **line_props)
#ax[1].axhline(y=1, color='black', label=r'$N$', **line_props)
ax[1].plot(ks, 1/ω_upper, color=c_acoustic, label=r'$\omega_+$', **line_props)

ax_eig[0].axhline(y=0, color='black', linestyle='dashed')
ax_eig[1].axhline(y=0, color='black', linestyle='dashed')

# Labels
ax[0].set_ylabel(r'frequency $\omega/N$')
ax[0].set_ylim(0, 5.2)
ax[0].set_xscale('log')
ax[1].set_xlabel(r'wavenumber $k_x$')
ax[1].set_ylabel(r'period $N/\omega$')
ax[1].set_ylim(0, 5.2)
ax[1].set_xscale('log')
ax[1].xaxis.set_label_coords(0.5, -0.15)
ax[0].yaxis.set_label_coords(-0.06, 0.5)
ax[1].yaxis.set_label_coords(-0.06, 0.5)

ax_eig[0].set_ylabel(r'$\sqrt{\rho}w$')
ax_eig[1].set_ylabel(r'$\sqrt{\rho}w$')
ax_eig[1].text(0, 0.65, 'gravity waves\n'+r'($\omega \leq \omega_-$)', verticalalignment='center', multialignment='center', fontsize=9)
ax_eig[0].text(0, 0.65, 'acoustic waves\n'+r'($\omega > \omega_+$)', verticalalignment='center', multialignment='center', fontsize=9)
ax_eig[1].set_xlabel(r'height $z$')
ax_eig[0].set_ylim(-1.3, 1.3)
ax_eig[1].set_ylim(-1.3, 1.3)
ax_eig[1].xaxis.set_label_coords(0.5, -0.15)
ax_eig[0].yaxis.set_label_coords(-0.06, 0.5)
ax_eig[1].yaxis.set_label_coords(-0.06, 0.5)

# Legends
legend = ax[1].legend(ncol=2, loc='upper left', frameon=False, fontsize=7, handlelength=1.5)
legend.get_frame().set_linewidth(0.0)

legend = ax_eig[1].legend(frameon=False, title=r'period $N/\omega$', loc='lower left', ncol=2, fontsize=7)
plt.setp(legend.get_title(),fontsize=9)
for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())
legend = ax_eig[0].legend(frameon=False, title=r'frequency $\omega/N$', loc='lower left', ncol=2, fontsize=7)
plt.setp(legend.get_title(),fontsize=9)
for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

# Save
ax[0].text(-0.06, 1.0, "(a)", transform=ax[0].transAxes, fontsize=9, va='top', ha='right')
ax[1].text(-0.06, 1.0, "(b)", transform=ax[1].transAxes, fontsize=9, va='top', ha='right')
ax_eig[0].text(-0.06, 1.0, "(a)", transform=ax_eig[0].transAxes, fontsize=9, va='top', ha='right')
ax_eig[1].text(-0.06, 1.0, "(b)", transform=ax_eig[1].transAxes, fontsize=9, va='top', ha='right')

fig.savefig('fig_waves_spectrum.pdf')
fig_eig.savefig('fig_waves_eigenfunctions.pdf')
