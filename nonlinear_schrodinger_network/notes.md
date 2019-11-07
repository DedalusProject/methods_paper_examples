# Nonlinear Schrodinger Network  #

## Problem Statement ##

These scripts solve the nonlinear Schrodinger equation on a network.
It demonstrates the ability of Dedalus to solve complex-valued PDEs and to solve PDEs on networks of 1D segments, akin to a spectral element method.
The implementation demonstrates how the Dedalus equation interface can be programatically extended to include complex equations.

## Execution ##

1. To run the simulation, run `run_problems.sh`. The script should run in approximately 1 cpu-hrs.

2. After success, run `run_analysis.sh` to produce figure X in the methods paper (Burns et al. 2019).

3. Optionally, run `run_extras.sh` to produce frames for a video of the solution.


