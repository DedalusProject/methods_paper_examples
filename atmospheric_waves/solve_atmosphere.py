"""
Radiative atmosphere

Solves for an atmosphere in hydrostatic and thermal equilibrium when energy transport is provided by radiation under the Eddington approximation and using a Kramer-like opacity.  The radiative opacity κ depends on the density ρ and temperature T as:

    κ = κ_0 * ρ^a * T^b

The system is formulated in terms of lnρ and lnT, and the solution utilizes the NLBVP system of Dedalus.  The computed atmosphere is saved in an HDF5 file "atmosphere.h5".  This program also produces the plots of the atmosphere used in the methods paper, stored in a file "atmosphere_a#_b#_eps#_part1.pdf", where the values of the a, b and epsilon coefficients are part of the file name.

It should take approximately 30 seconds on 1 Skylake core.
"""

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from dedalus import public as de
import h5py
import time
import logging
logger = logging.getLogger(__name__)

matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)
plt.style.use('./methods_paper.mplstyle')

comm = MPI.COMM_WORLD

# Parameters
ncc_cutoff = 1e-8
tolerance = 1e-8
a = 1
b = 0
nz = 128
IC = 'isothermal' #'polytrope'
F = 1e-5 # set to zero for analytic atmosphere comparison

# Derived parameters from Barekat & Brandenburg 2014
m_poly = (3-b)/(1+a)

gamma = 5/3
m_ad = 1/(gamma-1)
m = m_poly
logger.info("m={}, m_ad = {}, m_poly=(3-{})/(1+{})={}".format(m, m_ad, b, a, m_poly))

ln_Teff = -2
f = 1/3
q = 2/3

# Old Q calculation
fudge_factor = 1.25
τ0 = 4*f*np.exp(-4*ln_Teff*fudge_factor) - q
ε = q/τ0
Q = 1-ε
print(Q)

# New Q calculation
fudge_factor = 1.0
τ0 = 4*f*np.exp(-4*ln_Teff*fudge_factor) - q
ε = q/τ0
c6 = np.exp(ln_Teff)**4 / 4 / f
c3_c1 = c6_c4 = (1 - (c6*q)**(1+(a-b)/4)) * (1+a) / (1 + (a-b)/4)
Q = (m+1) * c3_c1 / 4
print(Q)
c4 = c6/c6_c4
c1 = (m+1)
η = c1/c4
print(c4, c1, η)
Lz = 1.25

logger.info("Target atmosphere has ln_Teff = {} and τ0 = {:g} for ε = {:g}".format(ln_Teff, τ0, ε))

tau_0_BB14 = 4e-4*np.array([1e4,1e5,1e6,1e7])*5
F_over_cE_BB14 = 1/4*(np.array([26600, 16300,9300,5200])/38968)**4
Q_BB14 = tau_0_BB14*F_over_cE_BB14

# Domain
z_basis = de.Legendre('z', nz, interval=(0,Lz), dealias=2)
domain = de.Domain([z_basis], np.float64, comm=MPI.COMM_SELF)

# Problem
problem = de.NLBVP(domain, variables=['ln_T', 'ln_rho'], ncc_cutoff=ncc_cutoff)
problem.parameters['a'] = a
problem.parameters['b'] = b
problem.parameters['g'] = g = (m+1)
problem.parameters['Lz'] = Lz
problem.parameters['gamma'] = gamma
problem.parameters['ε'] = ε
problem.parameters['η'] = η
problem.parameters['Q'] = Q
problem.parameters['F'] = F
problem.parameters['lnT0'] = lnT0 = 0
problem.parameters['lnρ0'] = lnρ0 = m*lnT0
problem.substitutions['ρκ(ln_rho,ln_T)'] = "exp(ln_rho*(a+1)+ln_T*(b))"
problem.add_equation("dz(ln_T) = -Q*exp(ln_rho*(a+1)+ln_T*(b-4))")
problem.add_equation("dz(ln_T) + dz(ln_rho) = -g*(1+F*exp(a*ln_rho+b*ln_T))*exp(-ln_T)")
problem.add_bc("left(ln_T)   = lnT0")
problem.add_bc("left(ln_rho) = lnρ0")

# Initial guess
solver = problem.build_solver()
z = domain.grid(0, scales=domain.dealias)
z_diag = domain.grid(0, scales=1)
ln_T = solver.state['ln_T']
ln_rho = solver.state['ln_rho']
ln_T.set_scales(domain.dealias)
ln_rho.set_scales(domain.dealias)

grad_ln_rho = domain.new_field()
if IC == 'polytrope':
    import scipy.special as scp
    z_phot = 0.8
    φ = 0.5*(1-scp.erf((z-z_phot)/0.05))
    ln_T['g'] = np.log(1-np.minimum(z, 0.9))*φ + np.log(1-z_phot)*(1-φ)
    dln_T = domain.new_field()
    ln_T.differentiate('z', out=dln_T)
    grad_ln_rho.set_scales(domain.dealias)
    grad_ln_rho['g'] = -(m+1)*np.exp(-ln_T['g']) - dln_T['g']
    logger.info('z_phot = {}'.format(z_phot))
if IC =='isothermal':
    ln_T['g'] = lnT0
    grad_ln_rho['g'] = -g
grad_ln_rho.antidifferentiate('z',('left',lnρ0), out=ln_rho)

# Diagnostics
diagnostics = solver.evaluator.add_dictionary_handler(group='diagnostics')
diagnostics.add_task('1/gamma*dz(ln_T) - (gamma-1)/gamma*dz(ln_rho)',name='dsdz_Cp')
diagnostics.add_task('1/gamma*ln_T - (gamma-1)/gamma*ln_rho',name='s_Cp')
diagnostics.add_task('-ρκ(ln_rho,ln_T)*η', name='dτ')
diagnostics.add_task('dz(ln_T)/dz(ln_rho)', name='1/m')

# Iterations
pert = solver.perturbations.data
pert.fill(1+tolerance)
do_plot = True
solver.evaluator.evaluate_group("diagnostics")
if do_plot:
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

iter = 0
start_time = time.time()
try:
    while np.sum(np.abs(pert)) > tolerance and np.sum(np.abs(pert)) < 1e6:
        if do_plot:
            ax.plot(z, ln_T['g'], label='lnT')
            ax.set_ylabel("lnT")
            ax2.plot(z, ln_rho['g'], label='lnrho', linestyle='dashed')
            ax2.set_ylabel("lnrho")
        solver.newton_iteration()
        logger.info('Perturbation norm: {:g}'.format(np.sum(np.abs(pert))))
        logger.info('iterates:  lnρ [{:.3g},{:.3g}]; lnT [{:.3g},{:.3g}]'.format(ln_rho['g'][-1],ln_rho['g'][0], ln_T['g'][-1],ln_T['g'][0]))
        solver.evaluator.evaluate_group("diagnostics")
        iter += 1
except:
    raise

end_time = time.time()

if do_plot:
    ax.plot(z, ln_T['g'], label='lnT')
    ax.set_ylabel("lnT")
    ax2.plot(z, ln_rho['g'], label='lnrho', linestyle='dashed')
    ax2.set_ylabel("lnrho")
    ax.set_title("iterative convergence of atmosphere")
    plt.savefig("atmosphere_iterations.pdf")

logger.info("converged in {:d} iter and in {:g} seconds".format(iter, end_time-start_time))

brunt2 = domain.new_field()
brunt2['g'] = diagnostics['dsdz_Cp']['g']*g
brunt2.set_scales(domain.dealias, keep_data=True)

Cs = domain.new_field()
Cs_z = domain.new_field()
Cs_zz = domain.new_field()
Cs.set_scales(domain.dealias)
Cs['g'] = np.sqrt(gamma)*np.exp(ln_T['g']*0.5)
Cs.differentiate('z', out=Cs_z)
Cs_z.differentiate('z', out=Cs_zz)
ω_ac2 = domain.new_field()
ω_ac2.set_scales(domain.dealias)
ω_ac2['g'] = Cs['g']*Cs_zz['g'] + gamma**2*g**2/(4*Cs['g']**2)
ω_lamb2 = domain.new_field()
ω_lamb2.set_scales(domain.dealias)
ω_lamb2['g'] = 17.5**2*Cs['g']**2
ω_plus2 = domain.new_field()
ω_plus2.set_scales(domain.dealias)
ω_minus2 = domain.new_field()
ω_minus2.set_scales(domain.dealias)
ω_plus2['g'] = ω_lamb2['g'] + ω_ac2['g']
ω_minus2['g'] = brunt2['g']*ω_lamb2['g']/(ω_lamb2['g'] + ω_ac2['g'])

dtau = domain.new_field()
tau = domain.new_field()
tau.set_scales(domain.dealias)
dtau['g'] = diagnostics['dτ']['g']
dtau.antidifferentiate('z',('right',0), out=tau)
i_tau_23 = (np.abs(tau['g']-q)).argmin()
z_phot = z[i_tau_23]
logger.info('photosphere is near z = {} (index {})'.format(z_phot, i_tau_23))
ln_T_phot = ln_T['g'][i_tau_23]
ln_T_top = ln_T.interpolate(z=Lz)['g'][0]
logger.info('ln_T_phot = {:.3g}, ln_T_top = {:.3g} and ln_T_top - ln_T_phot = {:.3g}'.format(ln_T_phot, ln_T_top, ln_T_top-ln_T_phot))
logger.info('T_phot = {:.3g}, T_top = {:.3g} and T_top/T_phot = {:.3g}'.format(np.exp(ln_T_phot), np.exp(ln_T_top), np.exp(ln_T_top-ln_T_phot)))

ln_rho_bot = ln_rho.interpolate(z=0)['g'][0]
ln_rho_top = ln_rho.interpolate(z=Lz)['g'][0]
logger.info("n_rho = {:.3g}".format(ln_rho_bot - ln_rho_top))

# Plot structure
fig = plt.figure(figsize=(3.4, 2.5))
ax1 = fig.add_subplot(2,1,1)
ax1.plot(z, ln_T['g'], label=r'$\ln \,T$', c='C0')
ax1.plot(z_phot, ln_T['g'][i_tau_23], marker='.', color='black')
ax1.set_ylabel(r"$\ln \,T$")
ax2 = ax1.twinx()
ax2.plot(z, ln_rho['g'], label=r'$\ln\,\rho$', linestyle='dashed', c='C1')
ax2.plot(z, ln_rho['g']+ln_T['g'], label=r'$\ln \,P$', linestyle='dashed', c='C2')
ax2.set_ylabel(r"$\ln \,\rho, \ln \,P$")
ax1.set_xlim([0, Lz])
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='lower left', frameon=False, fontsize=8, handlelength=1.8)
plt.setp(ax1.get_xticklabels(), visible=False)

ax = fig.add_subplot(2,1,2, sharex=ax1)
#ax.plot(z_diag, diagnostics['s_Cp']['g'], color='black', label=r'$s/c_P$')
#ax.set_ylabel(r"$s/c_P$")
#ax2 = ax.twinx()
max_N2 = np.max(brunt2['g'])
#ax.plot(z, np.sqrt(ω_ac2['g']/max_N2), color='darkblue', linestyle='dashed', label=r'$\omega_\mathrm{ac}$')
#ax.plot(z, np.sqrt(ω_lamb2['g']/max_N2), color='seagreen', linestyle='dashed', label=r'$\omega_\mathrm{L}$')
ax.plot(z, np.sqrt(ω_plus2['g']/max_N2), color='C0', label=r'$\omega_+$')
#ax.plot(z_phot, np.sqrt(brunt2['g'][i_tau_23]), marker='o', color='black', alpha=70)
ax.plot(z, np.sqrt(ω_minus2['g']/max_N2), color='C1', label=r'$\omega_-$')
ax.plot(z, np.sqrt(brunt2['g']/max_N2), color='black', linestyle='dashed', label=r'$N$')
ax.fill_between(z, np.sqrt(ω_plus2['g']/max_N2), y2=np.max(np.sqrt(ω_lamb2['g']/max_N2))+1, color='C0', alpha=0.3)
ax.fill_between(z, np.sqrt(ω_minus2['g']/max_N2), y2=-1, color='C1', alpha=0.3)
ax.set_ylim([-0.5, 5.5])
ax.set_ylabel(r'$\omega/N_{\mathrm{max}}$')
ax.set_xlabel(r'height $z$')
lines1, labels1 = ax.get_legend_handles_labels()
legend=ax.legend(lines1, labels1, loc='center left', frameon=False, ncol=1, fontsize=8, handlelength=1.8)
legend.get_frame().set_linewidth(0.0)
plt.tight_layout(pad=0.5)
fig.savefig('fig_waves_atmosphere.pdf')

# Plot T-P
width = 6
fig = plt.figure(figsize=(width, width/1.6*0.5))
ax = fig.add_subplot(1,1,1)
ln_P = ln_rho['g']+ln_T['g']
ax.plot(ln_P, ln_T['g'], label=r'$\ln T$')
ln_T_top_analytic = 1/(4*fudge_factor)*np.log(ε/(1+ε))
print("ln_T_top: {:g} and analytic {:g}".format(ln_T_top, ln_T_top_analytic))
ln_T_analytic = np.log((Q*np.exp(ln_P*(1+a)) + np.exp(ln_T_top_analytic*(4+1-b)))**(1/(4+a-b)))
ax.plot(ln_P, ln_T_analytic, linestyle='dashed', label=r'$\ln T_\mathrm{analytic}$')

ax.set_ylabel(r'$\ln T$')
ax.set_xlabel(r'$\ln P$')
ax.legend(frameon=False)
plt.tight_layout()
fig.savefig('atmosphere_eos.pdf')

error = domain.new_field()
error.set_scales(domain.dealias)
error['g'] = (ln_T['g']-ln_T_analytic)**2
print("L2 norm between calculated and analytic solution {:g} (F={:g})".format(np.sqrt(error.integrate('z')['g'][0]),F))

# Plot optical depth
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
ax.plot(z, tau['g'], label='τ')
ax.plot(z_phot, tau['g'][i_tau_23], marker='o')
ax.set_yscale('log')
ax.set_ylabel('τ(z)')
ax2 = ax.twinx()
ax2.plot(z, tau['g'], linestyle='dashed')
ax = fig.add_subplot(2,1,2)
ax.plot(ln_rho['g']+ln_T['g'], ln_T['g'], label=r'$\ln T$', linestyle='dashed')
lnP = np.linspace(-12, 0)

ln_T_top_analytic = 1/(4*fudge_factor)*np.log(ε/(1+ε))
print("ln_T_top: {:g} and analytic {:g}".format(ln_T_top, ln_T_top_analytic))
ax.plot(lnP,
        np.log((Q*np.exp(lnP*(1+a)) + np.exp(ln_T_top_analytic*(4+1-b)))**(1/(4+a-b))),
        linestyle='dashed')
ax.plot(lnP,
        np.log((Q/(m+1)*(1+a)/(4+a-b)*np.exp(lnP*(1+a)))**(1/(4+a-b))), linestyle='dotted')

ax.set_ylabel(r'$\ln T$')
ax.set_xlabel(r'$\ln P$')
ax2 = ax.twinx()
one_over_m = domain.new_field()
one_over_m['g'] = diagnostics['1/m']['g']
one_over_m.set_scales(domain.dealias, keep_data=True)
ax2.plot(ln_rho['g']+ln_T['g'], one_over_m['g'])
ax2.axhline(y=1/m, linestyle='dashed', color='black')
ax2.set_ylabel('1/m')
plt.savefig('atmosphere_optical_depth.pdf')

# Save structure
use_evaluator = False
if use_evaluator:
    atmosphere = solver.evaluator.add_file_handler('./atm')
    atmosphere.add_system(solver.state, layout='g')
    #solver.evaluator.evaluate_group('atmosphere')
    solver.evaluator.evaluate_handlers([atmosphere], world_time=0, wall_time=0, sim_time=0, timestep=0, iteration=0)

z_atmosphere = domain.grid(0, scales=1)
ln_T.set_scales(1, keep_data=True)
ln_rho.set_scales(1, keep_data=True)
atmosphere_file = h5py.File('./atmosphere.h5', 'w')
atmosphere_file['g'] = g
atmosphere_file['Lz'] = Lz
atmosphere_file['nz'] = nz
atmosphere_file['gamma'] = gamma
atmosphere_file['z'] = z_atmosphere
atmosphere_file['ln_T'] = ln_T['g']
atmosphere_file['ln_rho'] = ln_rho['g']
atmosphere_file['brunt_squared'] = brunt2['g']
atmosphere_file['c_s_squared'] = Cs['g']**2
atmosphere_file['ω_ac_squared'] = ω_ac2['g']
atmosphere_file.close()
