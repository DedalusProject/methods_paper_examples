import dedalus.public as de
from dedalus.tools  import post

import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

# parameters
nr = 512
nθ = 512
R_in = 1
R_out = 2

ν = 10.
κ = 0.01
ampl = 1.
delta = 0.02
nrot = 8 # four turns clockwise, four turns counterclockwise

# inner rotation periods
T_inner = 2*np.pi*R_in/ampl

r = de.Chebyshev('r',nr, interval=[R_in, R_out], dealias=3/2)
θ = de.Fourier('θ',nθ, dealias=3/2)
domain = de.Domain([θ, r], grid_dtype='float')

# forcing 
def SquareBoundaryForcing(*args):
    """square-wave in time forcing for left theta velocity

    """
    t = args[0].value # time
    ampl = args[1].value # max velocity
    delta = args[2].value # arctan width
    nrot = args[3].value # total number of rotations
    time_signal = (ampl/np.arctan(1/delta))*np.arctan(np.sin(t/nrot)/delta)
    return time_signal

def Forcing(*args, domain=domain, F=SquareBoundaryForcing):
    return de.operators.GeneralFunction(domain, layout='g', func=F, args=args)

de.operators.parseables['Vr'] = Forcing

problem = de.IVP(domain, variables=['φ','u', 'v', 'p', 'ur', 'vr', 'c1', 'c2', 'c3'])#, 'c1r', 'c2r', 'c3r'])

problem.parameters['ν'] = ν
problem.parameters['κ'] = κ
problem.parameters['ampl'] = ampl
problem.parameters['delta'] = delta
problem.parameters['nrot'] = nrot
problem.parameters['R_in'] = R_in

# not pre-multiplied...don't use this in an equation!
problem.substitutions['DivU'] = "ur + u/r + dθ(v)/r"
# assume pre-multiplication by r*r
problem.substitutions['Lap_s(f, f_r)'] = "r*r*dr(f_r) + r*f_r + dθ(dθ(f))"
problem.substitutions['Lap_r'] = "Lap_s(u, ur) - u - 2*dθ(v)"
problem.substitutions['Lap_θ'] = "Lap_s(v, vr) - v + 2*dθ(u)"
problem.substitutions['UdotGrad_s(f, f_r)'] = "r*r*u*f_r + r*v*dθ(f)"
#problem.substitutions['UdotGrad_r'] = "UdotGrad_s(u, ur) - r*v*v"
#problem.substitutions['UdotGrad_t'] = "UdotGrad_s(v, vr) + r*u*v"

# Stokes Flow
problem.add_equation('r*r*dr(p) - ν*Lap_r = 0')
problem.add_equation('r*r*dθ(p) - ν*Lap_θ = 0')

# Incompressibility
problem.add_equation('r*ur + u + dθ(v) = 0')

# dye equations
# problem.add_equation('r*r*dt(c1) - κ*Lap_s(c1, c1r) = -UdotGrad_s(c1, c1r)')
# problem.add_equation('r*r*dt(c2) - κ*Lap_s(c2, c2r) = -UdotGrad_s(c2, c2r)')
# problem.add_equation('r*r*dt(c3) - κ*Lap_s(c3, c3r) = -UdotGrad_s(c3, c3r)')
problem.add_equation('r*r*dt(c1) = -r*r*u*dr(c1) - r*v*dθ(c1)')
problem.add_equation('r*r*dt(c2) = -r*r*u*dr(c2) - r*v*dθ(c2)')
problem.add_equation('r*r*dt(c3) = -r*r*u*dr(c3) - r*v*dθ(c3)')

# First order
problem.add_equation('dr(u) - ur = 0')
problem.add_equation('dr(v) - vr = 0')
# problem.add_equation('dr(c1) - c1r = 0')
# problem.add_equation('dr(c2) - c2r = 0')
# problem.add_equation('dr(c3) - c3r = 0')

# Phi is just the rotational phase of the inner cylinder...solving as a PDE for convenience
problem.add_equation('dt(φ) - v/R_in = 0')

# boundary conditions
#problem.add_bc('left(v) = ampl')
problem.add_bc('left(v) = left(Vr(t, ampl, delta, nrot))')
problem.add_bc('right(v) = 0')

problem.add_bc('left(u) = 0')
problem.add_bc('right(u) = 0', condition="nθ != 0")
problem.add_bc('right(p) = 0', condition="nθ == 0")
# problem.add_bc('left(c1) = 0')
# problem.add_bc('right(c1) = 0')
# problem.add_bc('left(c2) = 0')
# problem.add_bc('right(c2) = 0')
# problem.add_bc('left(c3) = 0')
# problem.add_bc('right(c3) = 0')

solver = problem.build_solver(de.timesteppers.RK443)

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
snap.add_task('c1')#, scales=10)
snap.add_task('c2')#, scales=10)
snap.add_task('c3')#, scales=10)
snap.add_task('v')
snap.add_task('left(φ)')
analysis_tasks.append(snap)

# run control
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf
solver.stop_sim_time = nrot*T_inner
dt = 0.005

start  = time.time()
while solver.ok:
    if (solver.iteration-1) % 10 == 0:
        logger.info("Step {:d}; Time = {:e}".format(solver.iteration, solver.sim_time))
    solver.step(dt)
stop = time.time()

logger.info("Total Run time: {:5.2f} sec".format(stop-start))
logger.info('beginning join operation')
for task in analysis_tasks:
    logger.info(task.base_path)
    post.merge_analysis(task.base_path)
