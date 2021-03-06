import dedalus.public as de
from dedalus.tools  import post

import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

## Nondimensionalize using
# U --> inner cylinder velocity
# L --> gap width R_2 - R_1
#
# This leaves one parameter, Pe, the Peclet number associated with mass diffusion

# parameters
nr = 512
nθ = 512
Pe = 1e7 # if this is set to np.inf, mass diffusion will be switched off
if Pe is np.inf:
    finite_pe = False
else:
    finite_pe = True

# Geometry
R_in = 1
R_out = 2

logger.info("nr: {:d} nθ: {:d}".format(nr, nθ))
logger.info("Peclet number for diffusion = {:e}".format(Pe))
if finite_pe is False:
    logger.info("Diffision is off.")

delta = 0.02
nrot = 8 # four turns clockwise, four turns counterclockwise

# inner rotation periods
T_inner = 2*np.pi*R_in

# run control
# exploit fixed timesteps to ensure integer multiples of inner cylinder period

t_res = 1200.

dt = T_inner/t_res
n_step = nrot * t_res

r = de.Chebyshev('r',nr, interval=[R_in, R_out], dealias=3/2)
θ = de.Fourier('θ',nθ, dealias=3/2)
domain = de.Domain([θ, r], grid_dtype='float')

# forcing 
def SquareBoundaryForcing(*args):
    """square-wave in time forcing for left theta velocity

    """
    t = args[0].value # time
    delta = args[1].value # arctan width
    nrot = args[2].value # total number of rotations
    time_signal = (1./np.arctan(1/delta))*np.arctan(np.sin(t/nrot)/delta)
    return time_signal

def Forcing(*args, domain=domain, F=SquareBoundaryForcing):
    return de.operators.GeneralFunction(domain, layout='g', func=F, args=args)

de.operators.parseables['Vr'] = Forcing

variables = ['φ','u', 'v', 'p', 'ur', 'vr', 'c1', 'c2', 'c3']
if finite_pe:
    variables += ['c1r', 'c2r', 'c3r']
problem = de.IVP(domain, variables=variables)

problem.parameters['Pe'] = Pe
problem.parameters['delta'] = delta
problem.parameters['nrot'] = nrot
problem.parameters['R_in'] = R_in

# assume pre-multiplication by r*r
problem.substitutions['Lap_s(f, f_r)'] = "r*r*dr(f_r) + r*f_r + dθ(dθ(f))"
problem.substitutions['Lap_r'] = "Lap_s(u, ur) - u - 2*dθ(v)"
problem.substitutions['Lap_θ'] = "Lap_s(v, vr) - v + 2*dθ(u)"
problem.substitutions['UdotGrad_s(f, f_r)'] = "r*r*u*f_r + r*v*dθ(f)"

# Stokes Flow
problem.add_equation('r*r*dr(p) - Lap_r = 0')
problem.add_equation('r*r*dθ(p) - Lap_θ = 0')

# Incompressibility
problem.add_equation('r*ur + u + dθ(v) = 0')

# dye equations
if finite_pe:
    dye1_eqn = 'r*r*dt(c1) - Lap_s(c1, c1r)/Pe = -UdotGrad_s(c1, c1r)'
    dye2_eqn = 'r*r*dt(c2) - Lap_s(c2, c2r)/Pe = -UdotGrad_s(c2, c2r)'
    dye3_eqn = 'r*r*dt(c3) - Lap_s(c3, c3r)/Pe = -UdotGrad_s(c3, c3r)'
else:
    dye1_eqn = 'r*r*dt(c1) = -UdotGrad_s(c1, dr(c1))'
    dye2_eqn = 'r*r*dt(c2) = -UdotGrad_s(c2, dr(c2))'
    dye3_eqn = 'r*r*dt(c3) = -UdotGrad_s(c3, dr(c3))'
problem.add_equation(dye1_eqn)
problem.add_equation(dye2_eqn)
problem.add_equation(dye3_eqn)

# First order
problem.add_equation('dr(u) - ur = 0')
problem.add_equation('dr(v) - vr = 0')
if finite_pe:
    problem.add_equation('dr(c1) - c1r = 0')
    problem.add_equation('dr(c2) - c2r = 0')
    problem.add_equation('dr(c3) - c3r = 0')

# Phi is just the rotational phase of the inner cylinder...solving as a PDE for convenience
problem.add_equation('dt(φ) - v/R_in = 0')

# boundary conditions
problem.add_bc('left(v) = left(Vr(t, delta, nrot))')
problem.add_bc('right(v) = 0')

problem.add_bc('left(u) = 0')
problem.add_bc('right(u) = 0', condition="nθ != 0")
problem.add_bc('right(p) = 0', condition="nθ == 0")

if finite_pe:
    problem.add_bc('left(c1r) = 0')
    problem.add_bc('right(c1r) = 0')
    problem.add_bc('left(c2r) = 0')
    problem.add_bc('right(c2r) = 0')
    problem.add_bc('left(c3r) = 0')
    problem.add_bc('right(c3r) = 0')

# build solver and set stop times
solver = problem.build_solver(de.timesteppers.RK443)
solver.stop_wall_time = np.inf
solver.stop_iteration = n_step
solver.stop_sim_time = np.inf

# Initial conditions
def exp_circle(r,θ, r0, θ0, radius, ampl):
    logger.info("r0 = {}".format(r0))
    logger.info("θ0 = {}".format(θ0))
    xprime = r*np.cos(θ) - r0*np.cos(θ0)
    yprime = r*np.sin(θ) - r0*np.sin(θ0)
    rprime2 = xprime**2 + yprime**2
    
    return ampl*np.exp(-rprime2/radius**2)

c1 = solver.state['c1']
c2 = solver.state['c2']
c3 = solver.state['c3']

r0 = R_in + 0.5*(R_out - R_in) # place dye in middle of channel
radius = 0.25
c_ampl = 1.
c1_angle = 0
c2_angle = 2*np.pi/3.
c3_angle = 4*np.pi/3.

θθ, rr = domain.grids(scales=domain.dealias)

for f in [c1, c2, c3]:
    f.set_scales(domain.dealias, keep_data=False)
c1['g'] = exp_circle(rr, θθ, r0, c1_angle, radius, c_ampl)
c2['g'] = exp_circle(rr, θθ, r0, c2_angle, radius, c_ampl)
c3['g'] = exp_circle(rr, θθ, r0, c3_angle, radius, c_ampl)

# Analysis
analysis_tasks = []
check = solver.evaluator.add_file_handler('checkpoints', wall_dt=3540, max_writes=50)
check.add_system(solver.state)
analysis_tasks.append(check)

snap = solver.evaluator.add_file_handler('snapshots', sim_dt=T_inner/100, max_writes=50)
snap.add_task('c1')
snap.add_task('c2')
snap.add_task('c3')
snap.add_task('v')
snap.add_task('left(φ)') # save only phase at inner cylinder for visualization
analysis_tasks.append(snap)

start  = time.time()
logger.info("dt = {:e}".format(dt))
while solver.ok:
    if (solver.iteration-1) % 10 == 0:
        logger.info("Step {:d}; Time = {:e}".format(solver.iteration, solver.sim_time))
    solver.step(dt)
stop = time.time()

# write last checkpoint to ensure we return to zero inner cylinder phase
solver.evaluate_handlers_now(dt)

logger.info("Total Run time: {:5.2f} sec".format(stop-start))
logger.info('beginning join operation')
for task in analysis_tasks:
    logger.info(task.base_path)
    post.merge_analysis(task.base_path)
