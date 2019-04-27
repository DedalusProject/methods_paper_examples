import numpy as np
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt

from volume_penalty import Body, Ellipse


# Time-Space Parameters
dt     =  1e-3/4
Lx, Ly =  4., 4.
nx, ny =  512, 512

# Initial body angle; inital x,y = domain centre, not moving.
theta0 = np.pi/6

# Physical Parameters

mass    = 0.1 # mass
gravity = 1   # gravity

a,b = 2, 1 # ellipse semimajor/semiminor axes
eps = 0.025 # smoothing parameter

k    = 1   # boundary wavenumber
Bmax = 20 # boundary amplitude

eta =  10    # free magnetic diffusivity
chi = -0.1   # magnetic susceptibility
rho =  0.01 #5e-3  # electrical resistivity

# Create bases and domain
x_basis =   de.Fourier('x',nx, interval=(-Lx, Lx), dealias=3/2)
y_basis = de.Chebyshev('y',ny, interval=(-Ly, Ly), dealias=3/2)
domain  = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

# Create solid body
X = Body(Ellipse(a,b,eps),domain,angle=('theta',theta0))

Rx,Ry,Vx,Vy,M,Mx,My = domain.new_fields(7)

def set_fields():
    
    for f in [Rx,Ry,Vx,Vy,M,Mx,My]:
        f.set_scales(domain.dealias)
    
    Rx['g'], Ry['g'] = X.field(('x', 'y'), domain.dealias)
    Vx['g'], Vy['g'] = X.field(('vx','vy'),domain.dealias)
    Mx['g'], My['g'] = X.field('boundary', domain.dealias)

    M['g']   = X.field('interior', domain.dealias)
    Vx['g'] *= M['g']
    Vy['g'] *= M['g']


set_fields()


# Create inital and fixed boundary field

def Harmonic(amplitude,wavennumber,phase=0):
    a,b = domain.bases[0].interval
    k = 2*np.pi*wavennumber/(b-a)
    A = amplitude/k
    x,y = domain.grids(domain.dealias)
    y0  = domain.bases[1].interval[0]
    return -A*np.exp(-k*(y-y0))*np.sin(k*x+phase)

def show(A,M,size=6,lw=4.5,levels=12,color='Purples',name=None):
    
    A.set_scales(domain.dealias)
    M.set_scales(domain.dealias)
    x, y = domain.grids(domain.dealias)
    
    plt.figure(figsize=(size,size*Ly/Lx))
    plt.pcolormesh(x+0*y,y+0*x,(M['g']-np.min(M['g'])),cmap=color)
    plt.contour(x+0*y,y+0*x,A['g'],
                levels=np.linspace(-Bmax,Bmax,levels),
                colors=['black'],
                linewidths=[lw],
                linestyles=['solid'])
    plt.axis('off');
                
    if name != None:
        plt.savefig(name)


A0 = domain.new_field()
A0.set_scales(domain.dealias)
A0['g'] = Harmonic(Bmax,k)

show(A0,M,name='initial.png')

# 2D induction equation
problem = de.IVP(domain, variables=['A','Bx'])

# Vertical magnetic field, Electric field
problem.substitutions['By'] = "-dx(A)"
problem.substitutions['Ez'] = "Bx*(Vy + eta*rho*chi*My) - By*(Vx - eta*rho*chi*My)"

# Electromagnetic parameters
problem.parameters['eta']  = eta
problem.parameters['rho']  = rho
problem.parameters['chi']  = chi
problem.parameters['A0']   = A0

# Body-centred coordinates
problem.parameters['Rx'] = Rx
problem.parameters['Ry'] = Ry

# Eulerian solid-body velocity.
problem.parameters['Vx'] = Vx
problem.parameters['Vy'] = Vy

# Mask field and gradient
problem.parameters['M']  = M
problem.parameters['Mx'] = Mx
problem.parameters['My'] = My

# Horizontal magnetic field, Induction equation
problem.add_equation(" Bx - dy(A) = 0 ")
problem.add_equation(" dt(A) + eta*(dx(By) - dy(Bx)) = -Ez + eta*M*(1-rho)*(dx(By) - dy(Bx))")

# Fixed-field bottom, Potential-field top
problem.add_bc('left(A)  = left(A0)')
problem.add_bc('right(Bx - Hx(By)) = 0')

# Build solver
solver = problem.build_solver(de.timesteppers.SBDF2)
logger.info('Solver built')

# Initial conditions
A, Bx  = solver.state['A'], solver.state['Bx']
A.set_scales(domain.dealias)
Bx.set_scales(domain.dealias)

A['g'] = A0['g']*(1-M['g']) # Initial expelled flux
A.differentiate('y',out=Bx);


# Integration parameters
hour = 60*60
solver.stop_wall_time = 1*hour
solver.stop_sim_time  = np.inf
solver.stop_iteration = 5000*4

logger_freq = 100
output_freq = 100

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', iter=output_freq, max_writes=50)
snapshots.add_task("A")

#### We need to output X[:] every time step. This is the body [x,y,vx,vy,theta,omega] ###

# Fluid force on object
force = flow_tools.GlobalFlowProperty(solver, cadence=1)
force.add_property("(Bx**2 + By**2)*Mx",            name='Fx')
force.add_property("(Bx**2 + By**2)*My",            name='Fy')
force.add_property("(Bx**2 + By**2)*(Rx*My-Ry*Mx)", name='Tz')

chim = chi/(2*mass)
chii = chim/X.inertia

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        solver.step(dt)
        
        fx  = -chim*force.volume_average('Fx')
        fy  = -chim*force.volume_average('Fy') - gravity
        tz  = -chii*force.volume_average('Tz')
        
        X.step(dt,(fx,fy),tz)
        set_fields()
        
        if (solver.iteration-1) % logger_freq == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('    (%7s,%7s,%7s) (%7s,%7s,%7s)'%('fx','fy','tz','x','y','theta'))
            logger.info('    (%7.3f,%7.3f,%7.3f) (%7.3f,%7.3f,%7.3f)'%(fx,fy,tz,X['x'],X['y'],X['theta']*180/np.pi))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    run_time = (time.time() - start_time)/hour
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f min.' %(60*run_time))
    logger.info('Run time: %f cpu-hr' %(run_time*domain.dist.comm_cart.size))

