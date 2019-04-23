"""
Gravity waves

Describe the problem briefly.

It should take approximately X hours on Y (Ivy Bridge/Haswell/Broadwell/Skylake/etc) cores.
"""
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from dedalus import public as de

import logging
logger = logging.getLogger(__name__)

matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

comm = MPI.COMM_WORLD

logger.info("solving for linear, ideal waves, neglecting viscous and thermal diffusion")
from eigentools import Eigenproblem
import h5py

atmosphere_file = h5py.File('./atmosphere.h5', 'r')
z_atmosphere = atmosphere_file['z'][:]
ln_T = atmosphere_file['ln_T'][:]
ln_rho = atmosphere_file['ln_rho'][:]
brunt2 = atmosphere_file['brunt_squared'][:]
nz = atmosphere_file['nz'][()]
Lz = atmosphere_file['Lz'][()]
gamma = atmosphere_file['gamma'][()]
atmosphere_file.close()

z_basis = de.Chebyshev('z', nz, interval=(0,Lz))

gamma = 5/3

domain_EVP = de.Domain([z_basis], comm=MPI.COMM_SELF)
waves = de.EVP(domain_EVP, ['u','w','T1','ln_rho1'], eigenvalue='omega')

T0 = domain_EVP.new_field()
T0_z = domain_EVP.new_field()
ln_rho0 = domain_EVP.new_field()
del_ln_rho0 = domain_EVP.new_field()

T0.set_scales(1)
T0_z.set_scales(1)
ln_rho0.set_scales(1)
del_ln_rho0.set_scales(1)
T0['g'].real = np.exp(ln_T)
T0.differentiate('z', out=T0_z)
ln_rho0['g'] = ln_rho
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

brunt = np.sqrt(np.max(brunt2))

EP = Eigenproblem(waves, sparse=False)
#ks = np.linspace(0.5,10.5,num=50)
ks = np.logspace(0,1, num=5)
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

    for ikk, ik in enumerate(EP.evalues_good_index):
        ω = freqs[i][ikk]

        if np.abs(ω.imag) < brunt and np.abs(ω.real) < 1e-3 and ω.imag > 1e-3:
            EP.solver.set_state(ik)
            w = EP.solver.state['w']
            ax2.plot(z, w['g'], label='{:g}'.format(ω.imag))
    ax2.legend(loc='upper left', prop={'size': 6})
    #ax2.axvline(x=z_phot, linestyle='dashed', color='black')

with h5py.File('wave_frequencies.h5','w') as outfile:
    outfile.create_dataset('grid',data=ks)
    for i, freq in enumerate(freqs):
        outfile.create_dataset('freq_{}'.format(i),data=freq)
    outfile.create_dataset('brunt', data=brunt)
plt.show()
