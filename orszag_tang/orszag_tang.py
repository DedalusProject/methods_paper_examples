
import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Ly = (1., 1.)
Prandtl_m = 1.
Prandtl = 1.
Rm = 1e4

gamma = 5/3
c_v = 1/(gamma-1)
c_p = gamma*c_v

# Create bases and domain
x_basis = de.Fourier('x', 4096, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', 4096, interval=(0, Ly), dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['u','v','T','logrho','A'])

problem.substitutions['bx'] = "dy(A)"
problem.substitutions['by'] = "-dx(A)"

problem.substitutions['p_m'] = "0.5*(dx(A)*dx(A) + dy(A)*dy(A))"
problem.substitutions['div_u'] = "dx(u) + dy(v)"

problem.substitutions['viscous_u_lhs'] = "nu*(- dx(dx(u)) - dy(dy(u)) - 1./3.*dx(div_u))"
problem.substitutions['viscous_v_lhs'] = "nu*(- dx(dx(v)) - dy(dy(v)) - 1./3.*dy(div_u))"

problem.substitutions['viscous_u_rhs'] = "nu*(dx(logrho)*dx(u) + dy(logrho)*dy(u) " + \
                                          "+ dx(u)*dx(logrho) + dx(v)*dy(logrho) - 2./3.*dx(logrho)*div_u)"
problem.substitutions['viscous_v_rhs'] = "nu*(dx(logrho)*dx(v) + dy(logrho)*dy(v) " + \
                                          "+ dy(u)*dx(logrho) + dy(v)*dy(logrho) - 2./3.*dy(logrho)*div_u)"

problem.substitutions['viscous_heating'] = " 2.*dx(u)**2 +    dy(u)**2 " + \
                                           "  + dx(v)**2 + 2.*dy(v)**2 " + \
                                           "  + 2.*dx(v)*dy(u) " + \
                                           "  - 2./3.*div_u**2"

problem.substitutions['ohmic_heating'] = " (dx(dx(A)) + dy(dy(A)))**2 "

problem.parameters['eta'] = 1/Rm
problem.parameters['nu'] = Prandtl_m/Rm
problem.parameters['chi'] = Prandtl_m/Prandtl/Rm
problem.parameters['gamma'] = gamma
problem.parameters['c_v'] = c_v
problem.parameters['T0'] = T0 = 1./gamma

problem.add_equation("dt(u) + dx(T) + T0*dx(logrho) + viscous_u_lhs = -T*dx(logrho) - (u*dx(u) + v*dy(u)) + viscous_u_rhs + (bx*dx(bx) + by*dy(bx) - dx(p_m))/exp(logrho)")
problem.add_equation("dt(v) + dy(T) + T0*dy(logrho) + viscous_v_lhs = -T*dy(logrho) - (u*dx(v) + v*dy(v)) + viscous_v_rhs + (bx*dx(by) + by*dy(by) - dy(p_m))/exp(logrho)")

problem.add_equation("dt(logrho) + div_u = - u*dx(logrho) - v*dy(logrho)")

problem.add_equation("dt(T) + (gamma-1)*T0*div_u - chi/c_v*dx(dx(T)) - chi/c_v*dy(dy(T))  = " +
                      " - u*dx(T) - v*dy(T) - (gamma-1)*T*div_u + chi/c_v*dx(T)*dx(logrho) + chi/c_v*dy(T)*dy(logrho)" +
                      " + nu/c_v*viscous_heating" +
                      " + eta/c_v*ohmic_heating")

problem.add_equation("dt(A) - eta*(dx(dx(A)) + dy(dy(A))) = u*by - v*bx")

# Build solver
solver = problem.build_solver(de.timesteppers.RK443)
logger.info('Solver built')

# Initial conditions
x = domain.grid(0)
y = domain.grid(1)
A = solver.state['A']
u = solver.state['u']
v = solver.state['v']
logrho = solver.state['logrho']

logrho['g'] = np.log(25/(36*np.pi))

u['g'] = - np.sin(2*np.pi*y)
v['g'] =   np.sin(2*np.pi*x)

B0 = 1/np.sqrt(4*np.pi)
A['g'] = B0*( np.cos(4*np.pi*x)/(4*np.pi) + np.cos(2*np.pi*y)/(2*np.pi) )

# Initial timestep
dt = 0.000025

# Integration parameters
solver.stop_sim_time = 1.001
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots_Re1e4_4096', sim_dt=0.01, max_writes=10)
snapshots.add_system(solver.state)
snapshots.add_task("bx",name="Bx")
snapshots.add_task("by",name="By")
snapshots.add_task("exp(logrho)",name="rho")
snapshots.add_task("T+T0",name="temp")

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=5, safety=0.6,
                     max_change=1.5, min_change=0.5, max_dt=0.05, threshold=0.05)
CFL.add_velocities(('u','v'))
CFL.add_velocities(('bx','by'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(u*u + v*v) / eta", name='Rm')
flow.add_property("sqrt(bx*bx+by*by) / eta", name='S')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max Rm = %f' %flow.max('Rm'))
            logger.info('Max S = %f' %flow.max('S'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
