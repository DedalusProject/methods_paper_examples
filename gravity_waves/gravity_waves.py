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
