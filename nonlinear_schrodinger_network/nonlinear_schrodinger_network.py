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

import numpy as np
import matplotlib.pyplot as plt
import shelve

from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits

import logging
logger = logging.getLogger(__name__)

# Parameters
Nx = 512
Lx = 10
stop_sim_time = 20
timestepper = de.timesteppers.SBDF2
dt = 1e-2

# Graph structure
# Edges; the verticies go from 0:(nv-1).
edges=np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],
                [1,3],[3,5],[5,1],
                [0,6],[1,6],[2,6],[3,6],[4,6],[5,6]],dtype=int)

nv, ne = np.max(edges.T)+1, len(edges)

# Bases and domain
x_basis = de.Chebyshev('x', Nx, interval=(0, Lx), dealias=2)
domain  = de.Domain([x_basis], np.complex128)

# Incidence matrix
I = np.zeros([nv,ne],dtype=int)
for e in range(ne):
    v0,v1 = edges[e][0], edges[e][1]
    I[v0][e], I[v1][e] = np.sign(v0-v1), np.sign(v1-v0)

# Make variables and parameters list from the edges.
def s(e): return str(edges[e,0])+str(edges[e,1])

variables = parameters = []
for e in range(ne):
    variables  =  variables+['u'+s(e),'v'+s(e)]
    parameters = parameters+['a'+s(e)]

# Problem
problem = de.IVP(domain, variables=variables)
problem.meta[:]['x']['dirichlet'] = True

problem.parameters['i'] = i = 1j
for e in range(ne):
    problem.parameters[parameters[e]] = 1 # keep it simple

# Template equations for each edge.
problem.substitutions["L(u,v,a)"] = "i*dt(u) + 0.5*a*dx(v)"
problem.substitutions["N(u)"]   = "-u*abs(u)**2"
problem.substitutions["P(u,v,a)"] = "v - a*dx(u)"

# Bulk equations
for e in range(ne):
    u, v, a = variables[2*e], variables[2*e+1], parameters[e]
    problem.add_equation("L("+u+","+v+","+a+")=N("+u+")")
    problem.add_equation("P("+u+","+v+","+a+")=0")

# Boundary conditions:
def bc(n,u):
    if n == -1: return  "left("+u+")"
    if n ==  1: return "right("+u+")"

# continuity
for v in range(nv):
    edge=np.where(abs(I[v])==1)[0]
    BC1=bc(I[v,edge[0]],"u"+s(edge[0]))
    for e in range(1,len(edge)):
        BC2=bc(I[v,edge[e]],"u"+s(edge[e]))
        problem.add_bc(BC1+"-"+BC2+"=0")

# energy conservation
for v in range(nv):
    edge=np.where(abs(I[v])==1)[0]
    BC="("+str(I[v,edge[0]])+")*"+bc(I[v,edge[0]],"v"+s(edge[0]))
    for e in range(1,len(edge)):
        BC=BC+"+("+str(I[v,edge[e]])+")*"+bc(I[v,edge[e]],"v"+s(edge[e]))
    problem.add_bc(BC+"=0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_wall_time = np.inf
solver.stop_sim_time  = stop_sim_time
solver.stop_iteration = np.inf

# Initial conditions
def soliton(b,c,k):
    return k*np.exp(i*c*k*(x-b))/np.cosh(k*(x-b))

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
while solver.ok:
    solver.step(dt)
    if solver.iteration % 10 == 0:
        for var in range(len(variables)):
            X[var].set_scales(scale, keep_data=True)
            X_list[var].append(np.abs(np.copy(X[var]['g']))**2)
        t_list.append(solver.sim_time)
    if solver.iteration % 10 == 0:
        logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

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

