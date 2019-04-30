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
Nx = 64
Lx = 1

# Initial conditions
soliton_edge = 53
soliton_k = 20
soliton_c = 2

# Temporal discretization
safety = 0.5
N_cross = 100
C = soliton_c * soliton_k
dt = safety * (1 / Nx) / C
stop_iteration = int(N_cross / C / dt)
timestepper = de.timesteppers.SBDF2

# Output
save_dense_iter = int(1/40 / C / dt)
save_sparse_iter = int(1/4 / C / dt)

# Load graph
with np.load('graph.npz') as graph:
    edges = graph['edges']
    lengths = graph['lengths']

# Variable name definitions as functions of edge number
str_edge = lambda ne: f"{edges[ne][0]}_{edges[ne][1]}"
str_u = lambda ne: f"u_{str_edge(ne)}"
str_ux = lambda ne: f"ux_{str_edge(ne)}"
str_L = lambda ne: f"L_{str_edge(ne)}"

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
v_edges = [(str_u(ne), str_ux(ne)) for ne in range(N_edge)]
variables = [v for v_edge in v_edges for v in v_edge]
problem = de.IVP(domain, variables=variables)
problem.meta[:]['x']['dirichlet'] = True
# Length parameters
for ne in range(N_edge):
    problem.parameters[str_L(ne)] = lengths[ne]
problem.substitutions["abs_sq(u)"] = "u * conj(u)"
# Interior equations: NLS and first-order reduction for each edge
for ne in range(N_edge):
    u = str_u(ne)
    ux = str_ux(ne)
    L = str_L(ne)
    problem.add_equation(f"1j*dt({u}) + 0.5*dx({ux})/{L} = - {u}*abs_sq({u})")
    problem.add_equation(f"{ux} - dx({u})/{L} = 0")
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
solver.stop_sim_time  = np.inf
solver.stop_iteration = stop_iteration

# Initial conditions
def soliton(x, x0, c, k):
    return k * np.exp(1j*c*k*(x-x0)) / np.cosh(k*(x-x0))
x = domain.grid(0)
ui = solver.state.fields[2*soliton_edge]
uxi = solver.state.fields[2*soliton_edge+1]
Li = lengths[soliton_edge]
ui['g'] = soliton(Li*x, Li*Lx/2, soliton_c, soliton_k)
ui.differentiate('x', out=uxi)
uxi['g'] /= Li

# Outputs
s1 = solver.evaluator.add_file_handler('snap_dense', iter=save_dense_iter, max_writes=np.inf)
s2 = solver.evaluator.add_file_handler('snap_sparse', iter=save_sparse_iter, max_writes=np.inf)
for ne in range(N_edge):
    s1.add_task(str_u(ne), scales=domain.dealias)
    s2.add_task(str_u(ne), scales=domain.dealias)

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        solver.step(dt)
        if (solver.iteration - 1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))

