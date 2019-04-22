"""
Gravity waves

Describe the problem briefly.

It should take approximately X hours on Y (Ivy Bridge/Haswell/Broadwell/Skylake/etc) cores.
"""
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

comm = MPI.COMM_WORLD

ncc_cutoff = 1e-10
tolerance = 1e-10

a = 1
b = 0
b = -7/2
b = 1
#b = 1

nz = 512

# from Barekat & Brandenburg 2014
m_poly = (3-b)/(1+a)

gamma = 5/3
m_ad = 1/(gamma-1)
m = m_poly
logger.info("m={}, m_ad = {}, m_poly=(3-{})/(1+{})={}".format(m, m_ad, b, a, m_poly))
Lz = 0.25 ; F = 1.5 # works; bigger L or F doesn't
Lz = 2 ; F = 1.3 # works


z_basis = de.Chebyshev('z', nz, interval=(0,Lz), dealias=2)
domain = de.Domain([z_basis], np.float64, comm=MPI.COMM_SELF)

problem = de.NLBVP(domain, variables=['E','ln_T', 'ln_rho'], ncc_cutoff=ncc_cutoff)
problem.parameters['a'] = a
problem.parameters['b'] = b
problem.parameters['g'] = g = m+1
problem.parameters['Lz'] = Lz
problem.parameters['gamma'] = gamma
problem.parameters['F'] = F
problem.parameters['lnT0'] = lnT0 = 0
problem.parameters['lnρ0'] = lnρ0 = m*lnT0
problem.substitutions['ρκ(ln_rho,ln_T)'] = "exp(ln_rho*(a+1)+ln_T*(b))"
problem.add_equation("dz(E) = -3*F*ρκ(ln_rho,ln_T)")
problem.add_equation("E = exp(4*ln_T)")
problem.add_equation("dz(ln_T) + dz(ln_rho) = -g*exp(-ln_T)")
problem.add_bc("left(ln_T)   = lnT0")
problem.add_bc("left(ln_rho) = lnρ0")

# Setup initial guess
solver = problem.build_solver()
z = domain.grid(0, scales=domain.dealias)
z_diag = domain.grid(0, scales=1)
ln_T = solver.state['ln_T']
ln_rho = solver.state['ln_rho']
E = solver.state['E']
ln_T.set_scales(domain.dealias)
ln_rho.set_scales(domain.dealias)
E.set_scales(domain.dealias)
grad_ln_rho = domain.new_field()

#polytrope
#ln_T['g'] = np.log(1-z)
#ln_rho['g'] = m*ln_T['g']
#isothermal
ln_T['g'] = lnT0
grad_ln_rho['g'] = -g
grad_ln_rho.antidifferentiate('z',('left',lnρ0), out=ln_rho)

E['g'] = np.exp(4*ln_T['g'])

diagnostics = solver.evaluator.add_dictionary_handler(group='diagnostics')
diagnostics.add_task('1/gamma*dz(ln_T) - (gamma-1)/gamma*dz(ln_rho)',name='dsdz_Cp')
diagnostics.add_task('1/gamma*ln_T - (gamma-1)/gamma*ln_rho',name='s_Cp')
diagnostics.add_task('-ρκ(ln_rho,ln_T)', name='dτ')

# Iterations
pert = solver.perturbations.data
pert.fill(1+tolerance)
do_plot = True #True
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
        logger.info('Perturbation norm: {}'.format(np.sum(np.abs(pert))))
        logger.info('rho iterate:  {}--{}'.format(ln_rho['g'][-1],ln_rho['g'][0]))
        solver.evaluator.evaluate_group("diagnostics")
except:
    raise

ln_rho_bot = ln_rho.interpolate(z=0)['g'][0]
ln_rho_top = ln_rho.interpolate(z=Lz)['g'][0]
logger.info("n_rho = {}".format(ln_rho_bot - ln_rho_top))
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
ax.plot(z, ln_T['g'], label='T')
ax.set_ylabel(r"$\ln T$")
ax2 = ax.twinx()
ax2.plot(z, ln_rho['g'], label=r'$\ln \rho$', linestyle='dashed')
ax2.plot(z, ln_rho['g']+ln_T['g'], label=r'$\ln P$', linestyle='dashed')
ax2.set_ylabel(r"$\ln \rho, \ln P$")
ax = fig.add_subplot(2,1,2)
ax.plot(z_diag, diagnostics['s_Cp']['g'], label=r'$s/c_P$')
ax.plot(z_diag, diagnostics['dsdz_Cp']['g'], label=r'$\nabla s/c_P$')
ax.set_ylabel(r"$s/c_P$ and $\nabla s/c_P$")
ax.legend()

dtau = domain.new_field()
tau = domain.new_field()
tau.set_scales(domain.dealias)
dtau['g'] = diagnostics['dτ']['g'] # τ_c*np.exp(ln_rho['g']*(a+1)+ln_T['g']*b)
dtau.antidifferentiate('z',('right',0), out=tau)
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
ax.plot(z, tau['g'], label='τ')
ax.set_yscale('log')
ax.set_ylabel('τ(z)')
ax2 = ax.twinx()
ax2.plot(z, tau['g'], linestyle='dashed')
ax = fig.add_subplot(2,1,2)
ax.plot(ln_rho['g']+ln_T['g'], ln_T['g'], label=r'$\ln T$', linestyle='dashed')
ax.set_ylabel(r'$\ln T$')
ax.set_xlabel(r'$\ln P$')

i_tau_23 = (np.abs(tau['g']-2/3)).argmin()
logger.info(i_tau_23)
z_phot = z[i_tau_23]

logger.info('photosphere is at z = {}'.format(z_phot))

plt.show()
import sys
sys.exit()

logger.info("solving for linear, ideal waves, neglecting viscous and thermal diffusion")
from eigentools import Eigenproblem

domain_EVP = de.Domain([z_basis], comm=MPI.COMM_SELF)
waves = de.EVP(domain_EVP, ['u','w','T1','ln_rho1'], eigenvalue='omega')

T0 = domain_EVP.new_field()
T0_z = domain_EVP.new_field()
ln_rho0 = domain_EVP.new_field()
del_ln_rho0 = domain_EVP.new_field()

T0.set_scales(4)
T0_z.set_scales(4)
ln_rho0.set_scales(4)
del_ln_rho0.set_scales(4)
T0['g'].real = np.exp(ln_T['g'])
T0.differentiate('z', out=T0_z)
ln_rho0['g'] = np.exp(ln_rho['g'])
ln_rho0.differentiate('z', out=del_ln_rho0)
T0.set_scales(1, keep_data=True)
T0_z.set_scales(1, keep_data=True)
ln_rho0.set_scales(1, keep_data=True)
del_ln_rho0.set_scales(1, keep_data=True)

waves.parameters['T0'] = T0
waves.parameters['T0_z'] = T0_z
waves.parameters['del_ln_rho0'] = del_ln_rho0
waves.parameters['gamma'] = gamma
waves.parameters['k'] = 1
waves.substitutions['dt(A)'] = 'omega*A'
waves.substitutions['dx(A)'] = '1j*k*A'
waves.substitutions['Div_u'] = 'dx(u) + dz(w)'
logger.debug("Setting z-momentum equation")
waves.add_equation("dt(w) + dz(T1) + T0*dz(ln_rho1) + T1*del_ln_rho0 = 0 ")
logger.debug("Setting x-momentum equation")
waves.add_equation("dt(u) + dx(T1) + T0*dx(ln_rho1)                  = 0 ")
logger.debug("Setting continuity equation")
waves.add_equation("dt(ln_rho1) + w*del_ln_rho0 + Div_u  = 0 ")
logger.debug("Setting energy equation")
waves.add_equation("dt(T1) + w*T0_z + (gamma-1)*T0*Div_u = 0 ")
waves.add_bc('left(dz(u)) = 0')
waves.add_bc('right(dz(u)) = 0')
waves.add_bc('left(dz(T1)) = 0')

brunt = np.sqrt(diagnostics['dsdz_Cp']['g'][-1]*g)
logger.info('T0 = {}, N = {}'.format(T0['g'][0], brunt))

H = -1/del_ln_rho0['g'][-1]
other_brunt = np.sqrt((gamma-1)/gamma*g/H)
logger.info('other brunt: {} (H = {})'.format(other_brunt, H))


import h5py
EP = Eigenproblem(waves, sparse=False)
ks = np.linspace(0.5,10.5,num=50)
ks = [1,3,5]
freqs = []
for i, k in enumerate(ks):
    EP.EVP.namespace['k'].value = k
    EP.EVP.parameters['k'] = k
    EP.solve()
    EP.reject_spurious()
    y = EP.evalues_good
    freqs.append(y)
    fig2, ax2 = plt.subplots()
    z = domain_EVP.grid(0)
    #print('len freqs: {}'.format(len(freqs)))
    for ikk, ik in enumerate(EP.evalues_good_index):
        ω = freqs[i][ikk]
        #print(ik, ω, brunt)
        if np.abs(ω.imag) < brunt and np.abs(ω.real) < 1e-3 and ω.imag > 1e-3:
            print(ω, brunt)
            EP.solver.set_state(ik)
            w = EP.solver.state['w']
            ax2.plot(z, w['g'], label='{:g}'.format(ω.imag))
    ax2.legend(loc='upper left', prop={'size': 6})
    ax2.axvline(x=z_phot, linestyle='dashed', color='black')

with h5py.File('wave_frequencies.h5','w') as outfile:
    outfile.create_dataset('grid',data=ks)
    for i, freq in enumerate(freqs):
        outfile.create_dataset('freq_{}'.format(i),data=freq)

plt.show()
