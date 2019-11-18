"""
Gravity waves

Solves linearized, ideal, fully compressible wave problem to obtain frequencies and eigenfunmctions.  This solves the dense eigenvalue problem to obtain all "good" eigenvalue/eigenfunction pairs (at all "good" vertical wavenumbers), at a selection of different horizontal wavenumbes kx.  Eigentools is used to assess whether eigenvalues are "good".  Eigenfunctions are normalized under a kinetic energy weight and are rotated in a consistent fashion within the complex space.  Accepted eigenvalue/eigenfunction pairs are stored in HDF5 file "wave_frequencies.h5", along with various bits of atmosphere metadata that is helpful in later analysis.

It should take approximately 6 hours on 1 Skylake core.
"""

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from dedalus import public as de
import time
from eigentools import Eigenproblem
import h5py
import logging
logger = logging.getLogger(__name__)
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)


# Parameters
nz_waves = 384
ncc_cutoff = 1e-6

cw_size = MPI.COMM_WORLD.size
cw_rank = MPI.COMM_WORLD.rank

# Load background
atmosphere_file = h5py.File('./atmosphere.h5', 'r')
z_atmosphere = atmosphere_file['z'][:]
ln_T = atmosphere_file['ln_T'][:]
ln_rho = atmosphere_file['ln_rho'][:]
brunt2 = atmosphere_file['brunt_squared'][:]
ω_ac2 = atmosphere_file['ω_ac_squared'][:]
c_s2 = atmosphere_file['c_s_squared'][:]
nz = atmosphere_file['nz'][()]
Lz = atmosphere_file['Lz'][()]
gamma = atmosphere_file['gamma'][()]
atmosphere_file.close()

# Problem
gamma = 5/3
z_basis = de.Chebyshev('z', nz_waves, interval=(0, Lz), tau_after_pre=True)
domain_EVP = de.Domain([z_basis], comm=MPI.COMM_SELF)
waves = de.EVP(domain_EVP, ['u','w','T1','ln_rho1', 'w_z'], eigenvalue='omega', ncc_cutoff=ncc_cutoff)
n_var = 5
T0 = domain_EVP.new_field()
T0_z = domain_EVP.new_field()
ln_rho0 = domain_EVP.new_field()
del_ln_rho0 = domain_EVP.new_field()

# Resample background
scale = nz/nz_waves
logger.info("data nz ({:d}) is {} times larger than wave nz ({:d})".format(nz,scale,nz_waves))
T0.set_scales(scale)
T0_z.set_scales(scale)
ln_rho0.set_scales(scale)
del_ln_rho0.set_scales(scale)
T0['g'].real = np.exp(ln_T)
T0.differentiate('z', out=T0_z)
ln_rho0['g'] = ln_rho
ln_rho0.differentiate('z', out=del_ln_rho0)
T0.set_scales(1, keep_data=True)
T0_z.set_scales(1, keep_data=True)
ln_rho0.set_scales(1, keep_data=True)
del_ln_rho0.set_scales(1, keep_data=True)

z = domain_EVP.grid(0)
fig, ax = plt.subplots(nrows=2)
ax[0].plot(z, T0['g'])
ax[0].plot(z, ln_rho0['g'])
ax[1].plot(z, T0_z['g'])
ax[1].plot(z, del_ln_rho0['g'])

waves.parameters['T0'] = T0
waves.parameters['T0_z'] = T0_z
waves.parameters['del_ln_rho0'] = del_ln_rho0
waves.parameters['gamma'] = gamma
waves.parameters['k'] = 1
waves.substitutions['dt(A)'] = '1j*omega*A'
waves.substitutions['dx(A)'] = '-1j*k*A'
waves.substitutions['Div_u'] = 'dx(u) + w_z'
logger.debug("Setting z-momentum equation")
waves.add_equation("dt(w) + dz(T1) + T0*dz(ln_rho1) + T1*del_ln_rho0 = 0 ")
logger.debug("Setting x-momentum equation")
waves.add_equation("dt(u) + dx(T1) + T0*dx(ln_rho1)                  = 0 ")
logger.debug("Setting continuity equation")
waves.add_equation("dt(ln_rho1) + w*del_ln_rho0 + Div_u  = 0 ")
logger.debug("Setting energy equation")
waves.add_equation("dt(T1) + w*T0_z + (gamma-1)*T0*Div_u = 0 ")
waves.add_equation("dz(w) - w_z = 0 ")
#waves.add_bc('left(dz(u)) = 0')
#waves.add_bc('right(dz(u)) = 0')
waves.add_bc('left(w) = 0')
waves.add_bc('right(w) = 0')

# value at top of atmosphere in isothermal layer
brunt_max = np.max(np.sqrt(np.abs(brunt2))) # max value in atmosphere
k_Hρ = -1/2*del_ln_rho0.interpolate(z=0)['g'][0].real
c_s = np.sqrt(T0.interpolate(z=0)['g'][0].real)

logger.info("max Brunt is |N| = {} and  k_Hρ is {}".format(brunt_max, k_Hρ))
start_time = time.time()
EP = Eigenproblem(waves)

# Distribute wavenumbers
ks = np.logspace(-1, 2, num=20) * k_Hρ
batch = int(np.ceil(len(ks) / cw_size))
ks_local = ks[batch*cw_rank:batch*(cw_rank+1)]

freqs = []
eigenfunctions = {'w':[], 'u':[], 'T':[]}
omega = {'ω_plus_min':[], 'ω_minus_max':[]}
w_weights = []
KE = domain_EVP.new_field()
rho0 = domain_EVP.new_field()
rho0['g'] = np.exp(ln_rho0['g'])
rho0_avg = (rho0.integrate('z')['g'][0]/Lz).real
logger.debug("aveage ρ0 = {:g}".format(rho0_avg))
fig, ax = plt.subplots()

for i, k in enumerate(ks_local):
    ω_lamb2 = k**2*c_s2
    ω_plus2 = ω_lamb2 + ω_ac2
    ω_minus2  = brunt2*ω_lamb2/(ω_lamb2 + ω_ac2)
    omega['ω_plus_min'].append(np.min(np.sqrt(ω_plus2)))
    omega['ω_minus_max'].append(np.max(np.sqrt(ω_minus2)))
    EP.EVP.namespace['k'].value = k
    EP.EVP.parameters['k'] = k
    EP.solve()
    EP.reject_spurious()
    ω = EP.evalues_good
    ax.plot([k]*len(ω), np.abs(ω.real)/brunt_max, marker='x', linestyle='none')
    freqs.append(ω)
    eigenfunctions['w'].append([])
    eigenfunctions['u'].append([])
    eigenfunctions['T'].append([])
    w_weights.append([])
    logger.info("k={:g} ; {:d} good eigenvalues among {:d} fields ({:g}%)".format(k, EP.evalues_good_index.shape[0], n_var, EP.evalues_good_index.shape[0]/(n_var*nz_waves)*100))
    for ikk, ik in enumerate(EP.evalues_good_index):
        EP.solver.set_state(ik)
        w = EP.solver.state['w']
        u = EP.solver.state['u']
        T = EP.solver.state['T1']

        i_max = np.argmax(np.abs(w['g']))
        phase_correction = w['g'][i_max]
        w['g'] /= phase_correction
        u['g'] /= phase_correction
        T['g'] /= phase_correction

        KE['g'] = 0.5*rho0['g']*(u['g']*np.conj(u['g'])+w['g']*np.conj(w['g'])).real
        KE_avg = (KE.integrate('z')['g'][0]/Lz).real
        weight = np.sqrt(KE_avg/(0.5*rho0_avg))

        eigenfunctions['w'][i].append(np.copy(w['g'])/weight)
        eigenfunctions['u'][i].append(np.copy(u['g'])/weight)
        eigenfunctions['T'][i].append(np.copy(T['g'])/weight)

ax.set_xscale('log')
end_time = time.time()
logger.info("time to solve all modes: {:g} seconds".format(end_time-start_time))

# Gather to root node
def gather_concat(obj):
    obj = MPI.COMM_WORLD.gather(obj, root=0)
    if cw_rank == 0:
        return [item for rank_obj in obj for item in rank_obj]

freqs = gather_concat(freqs)
ef_w = gather_concat(eigenfunctions['w'])
ef_u = gather_concat(eigenfunctions['u'])
ef_T = gather_concat(eigenfunctions['T'])
eigenfunctions = {'w': ef_w, 'u': ef_u, 'T': ef_T}
opm = gather_concat(omega['ω_plus_min'])
omm = gather_concat(omega['ω_minus_max'])
omega = {'ω_plus_min': opm, 'ω_minus_max': omm}
w_weights = gather_concat(w_weights)

# Save data
if cw_rank == 0:
    with h5py.File('wave_frequencies.h5','w') as outfile:
        scale_group = outfile.create_group('scales')
        scale_group.create_dataset('grid',data=ks)
        scale_group.create_dataset('brunt_max', data=brunt_max)
        scale_group.create_dataset('k_Hρ',  data=k_Hρ)
        scale_group.create_dataset('c_s',   data=c_s)
        scale_group.create_dataset('z',   data=z)
        scale_group.create_dataset('Lz',  data=Lz)
        scale_group.create_dataset('rho0', data=rho0['g'])
        tasks_group = outfile.create_group('tasks')
        for i, freq in enumerate(freqs):
            data_group = tasks_group.create_group('k_{:03d}'.format(i))
            data_group.create_dataset('freq',data=freq)
            data_group.create_dataset('ω_plus_min',data=omega['ω_plus_min'][i])
            data_group.create_dataset('ω_minus_max',data=omega['ω_minus_max'][i])
            data_group.create_dataset('eig_w',data=eigenfunctions['w'][i])
            data_group.create_dataset('eig_u',data=eigenfunctions['u'][i])
            data_group.create_dataset('eig_T',data=eigenfunctions['T'][i])
