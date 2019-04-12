"""
Nonlinear Schrodinger equation on a network.

This scripts simulations the nonlinear Schrodinger equation on a network of
1-dimensional segments. It demonstrates the ability of Dedalus to simulate
complex-valued equations and the flexibility to mimic a spectral element
method by "connecting" separate domains via boundary conditions for different
fields definied on the same Dedalus domain.

This script should be ran serially, because it is in 1D. It should take
approximately X hours on Y (Ivy Bridge/Haswell/Broadwell/Skylake/etc) cores.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import shelve

from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits

import logging
logger = logging.getLogger(__name__)


################
## Parameters ##
################

# Spatial discretization
Nx = 512
Lx = 10

# Temporal discretization
dt = 1e-2
stop_sim_time = 20
timestepper = de.timesteppers.SBDF2
save_iter = 10

# Graph structure defined as list of directed edges
edges = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]]
edges += [[1,3],[3,5],[5,1]]
edges += [[0,6],[1,6],[2,6],[3,6],[4,6],[5,6]]
#edges = [[0,1], [1,2], [3,1]]

################
## Simulation ##
################

# Incidence matrix
edges = np.array(edges)
N_edge = edges.shape[0]
N_vert = 1 + np.max(edges)
I = np.zeros([N_vert, N_edge], dtype=int)
for ne, edge in enumerate(edges):
    v0, v1 = edge
    I[v0][ne] = -1
    I[v1][ne] = 1

# Bases and domain
x_basis = de.Chebyshev('x', Nx, interval=(0, Lx), dealias=2)
domain  = de.Domain([x_basis], np.complex128)

# Problem
# Make u and ux variables for each edge
str_edge = lambda ne: f"{edges[ne][0]}_{edges[ne][1]}"
str_u = lambda ne: f"u_{str_edge(ne)}"
str_ux = lambda ne: f"ux_{str_edge(ne)}"
variables = [str_u(ne) for ne in range(N_edge)] + [str_ux(ne) for ne in range(N_edge)]
problem = de.IVP(domain, variables=variables)
problem.substitutions["abs_sq(u)"] = "u * conj(u)"
# Interior equations: NLS and first-order reduction for each edge
for ne in range(N_edge):
    u = str_u(ne)
    ux = str_ux(ne)
    problem.add_equation(f"1j*dt({u}) + 0.5*dx({ux}) = - {u}*abs_sq({u})")
    problem.add_equation(f"{ux} - dx({u}) = 0")
# Boundary conditions: continuity for each vertex
def str_end(f, s):
    if s == -1: return f"left({f})"
    if s ==  1: return f"right({f})"
for nv in range(N_vert):
    # Get proper restriction of each incident edge value
    u_edges = [str_end(str_u(ne), s) for ne, s in enumerate(I[nv]) if s != 0]
    # Apply continuity to edges
    for ne in range(len(u_edges) - 1):
        un = u_edges[ne]
        um = u_edges[ne + 1]
        problem.add_bc(f"{un} - {um} = 0")
# Boundary conditions: Kirchoff's law at each vertex
for nv in range(N_vert):
    # Get proper restriction of each incident edge gradient
    ux_edges = [str_end(str_ux(ne), s) for ne, s in enumerate(I[nv]) if s != 0]
    s_edges = [s for s in I[nv] if s != 0]
    # Sum of signed gradients must be zero
    problem.add_bc(f"{'+'.join(f'({s})*{ux}' for s, ux in zip(s_edges, ux_edges))} = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_wall_time = np.inf
solver.stop_sim_time  = stop_sim_time
solver.stop_iteration = np.inf

# Initial conditions
def soliton(b,c,k):
    return k*np.exp(i*c*k*(x-b))/np.cosh(k*(x-b))

variables = problem.variables
x, scale = domain.grid(0), 1
X=X_list=[]
for var in range(len(variables)):
    X  =  X+[solver.state[variables[var]]]
    if variables[var]=="u06":
        X[var]['g'] = 1*soliton(Lx/2,3,2)
    if variables[var]=="v06":
        X[var-1].differentiate(0, out=X[var])
    X[var].set_scales(scale, keep_data=True)
    X_list = X_list+[ [np.abs(np.copy(X[var]['g']))**2] ]

# Main loop
t_list = [solver.sim_time]
# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        solver.step(dt)
        if (solver.iteration - 1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
        if (solver.iteration - 1) % save_iter == 0:
            for var in range(len(variables)):
                X[var].set_scales(scale, keep_data=True)
                X_list[var].append(np.abs(np.copy(X[var]['g']))**2)
            t_list.append(solver.sim_time)
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))






def edge_plot(ulist,tlist,title,fname):
    u_array = np.array(ulist)
    t_array = np.array(tlist)
    xmesh, ymesh = quad_mesh(x=x, y=t_array)
    plt.figure()
    plt.pcolormesh(xmesh, ymesh, u_array, cmap='RdBu_r')
    plt.axis(pad_limits(xmesh, ymesh))
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(title)
    plt.savefig(fname)

for e in range(len(edges)):
    edge_plot(X_list[2*e],t_list,"edge "+s(e),"./fano_plane/edge_"+s(e)+".png")

# Save output
filename = "edges.dat"
with shelve.open(filename, protocol=-1) as file:
    file['t'] = np.array(t_list)
    file['x'] = x
    for e in range(len(edges)):
        file['u{}'.format(s(e))] = np.array(X_list[2*e])

