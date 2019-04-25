"""
Radiative atmosphere

Solves for an atmosphere in hydrostatic and thermal equilibrium when energy transport is provided by radiation under the Eddington approximation and using a Kramer-like opacity.  The radiative opacity κ depends on the density ρ and temperature T as:

    κ = κ_0 * ρ^a * T^b

The system is formulated in terms of lnρ and lnT, and the solution utilizes the NLBVP system of Dedalus.  The computed atmosphere is saved in an HDF5 file "atmosphere.h5".

It should take approximately 3 seconds on 1 Haswell core.
"""
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from dedalus import public as de
import h5py

import logging
logger = logging.getLogger(__name__)

matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

comm = MPI.COMM_WORLD

ncc_cutoff = 1e-10
tolerance = 1e-8

a = 1
b = 0
#b = -7/2
#b = 1
#b = 1

nz = 512

# from Barekat & Brandenburg 2014
m_poly = (3-b)/(1+a)

gamma = 5/3
m_ad = 1/(gamma-1)
m = m_poly
logger.info("m={}, m_ad = {}, m_poly=(3-{})/(1+{})={}".format(m, m_ad, b, a, m_poly))
Lz = 0.2 ; Q = 1.5 # works; bigger L or F doesn't
ε = 1e-4
Lz = 2 ; Q = 4/3-ε  # Q=1.33 is the largest that works

tau_0_BB14 = 4e-4*np.array([1e4,1e5,1e6,1e7])*5
F_over_cE_BB14 = 1/4*(np.array([26600, 16300,9300,5200])/38968)**4
Q_BB14 = tau_0_BB14*F_over_cE_BB14

z_basis = de.Chebyshev('z', nz, interval=(0,Lz), dealias=2)
domain = de.Domain([z_basis], np.float64, comm=MPI.COMM_SELF)

problem = de.NLBVP(domain, variables=['ln_T', 'ln_rho'], ncc_cutoff=ncc_cutoff)
problem.parameters['a'] = a
problem.parameters['b'] = b
problem.parameters['g'] = g = (m+1)
problem.parameters['Lz'] = Lz
problem.parameters['gamma'] = gamma
problem.parameters['Q'] = Q
problem.parameters['lnT0'] = lnT0 = 0
problem.parameters['lnρ0'] = lnρ0 = m*lnT0
problem.substitutions['ρκ(ln_rho,ln_T)'] = "exp(ln_rho*(a+1)+ln_T*(b))"
problem.add_equation("4/3*dz(ln_T) = -Q*exp(ln_rho*(a+1)+ln_T*(b-4))")
problem.add_equation("dz(ln_T) + dz(ln_rho) = -g*exp(-ln_T)")
problem.add_bc("left(ln_T)   = lnT0")
problem.add_bc("left(ln_rho) = lnρ0")

# Setup initial guess
solver = problem.build_solver()
z = domain.grid(0, scales=domain.dealias)
z_diag = domain.grid(0, scales=1)
ln_T = solver.state['ln_T']
ln_rho = solver.state['ln_rho']
ln_T.set_scales(domain.dealias)
ln_rho.set_scales(domain.dealias)

IC = 'polytrope'
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

diagnostics = solver.evaluator.add_dictionary_handler(group='diagnostics')
diagnostics.add_task('1/gamma*dz(ln_T) - (gamma-1)/gamma*dz(ln_rho)',name='dsdz_Cp')
diagnostics.add_task('1/gamma*ln_T - (gamma-1)/gamma*ln_rho',name='s_Cp')
diagnostics.add_task('-ρκ(ln_rho,ln_T)', name='dτ')
diagnostics.add_task('dz(ln_T)/dz(ln_rho)', name='1/m')

# Iterations
pert = solver.perturbations.data
pert.fill(1+tolerance)
do_plot = True
solver.evaluator.evaluate_group("diagnostics")
if do_plot:
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

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
except:
    plt.show()
    raise

if do_plot:
    ax.plot(z, ln_T['g'], label='lnT')
    ax.set_ylabel("lnT")
    ax2.plot(z, ln_rho['g'], label='lnrho', linestyle='dashed')
    ax2.set_ylabel("lnrho")


brunt2 = domain.new_field()
brunt2['g'] = diagnostics['dsdz_Cp']['g']*g
brunt2.set_scales(domain.dealias, keep_data=True)

dtau = domain.new_field()
tau = domain.new_field()
tau.set_scales(domain.dealias)
dtau['g'] = diagnostics['dτ']['g']
dtau.antidifferentiate('z',('right',0), out=tau)
i_tau_23 = (np.abs(tau['g']-2/3)).argmin()
z_phot = z[i_tau_23]
logger.info('photosphere is near z = {} (index {})'.format(z_phot, i_tau_23))

ln_rho_bot = ln_rho.interpolate(z=0)['g'][0]
ln_rho_top = ln_rho.interpolate(z=Lz)['g'][0]
logger.info("n_rho = {}".format(ln_rho_bot - ln_rho_top))
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
ax.plot(z, ln_T['g'], label='T')
ax.set_ylabel(r"$\ln T$")
ax.plot(z_phot, ln_T['g'][i_tau_23], marker='o')
ax2 = ax.twinx()
ax2.plot(z, ln_rho['g'], label=r'$\ln \rho$', linestyle='dashed')
ax2.plot(z, ln_rho['g']+ln_T['g'], label=r'$\ln P$', linestyle='dashed')
ax2.set_ylabel(r"$\ln \rho, \ln P$")
ax = fig.add_subplot(2,1,2)
ax.plot(z_diag, diagnostics['s_Cp']['g'], label=r'$s/c_P$')
ax.plot(z_diag, diagnostics['dsdz_Cp']['g'], label=r'$\nabla s/c_P$')
ax.set_ylabel(r"$s/c_P$ and $\nabla s/c_P$")
ax2 = ax.twinx()
ax2.plot(z, brunt2['g'], color='black', linestyle='dashed')
ax2.plot(z_phot, brunt2['g'][i_tau_23], marker='o', color='black')
ax2.set_ylabel(r'$N^2$')
ax.legend()

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
ax.set_ylabel(r'$\ln T$')
ax.set_xlabel(r'$\ln P$')
ax2 = ax.twinx()
one_over_m = domain.new_field()
one_over_m['g'] = diagnostics['1/m']['g']
one_over_m.set_scales(domain.dealias, keep_data=True)
ax2.plot(ln_rho['g']+ln_T['g'], one_over_m['g'])
ax2.set_ylabel('1/m')

fig, ax = plt.subplots(nrows=2)
ax[0].plot(z, np.exp(ln_T['g']))
ax[1].plot(z, np.exp(ln_rho['g']))

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
atmosphere_file.close()

plt.show()
