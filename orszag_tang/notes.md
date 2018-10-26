# Orszag-Tang Vortex  #


## Problem Statement ##

The Orszag-Tang vortex is a classic test problem of compressible magnetohydrodynamics (MHD). The simulation is run in a 2D periodic domain. A supersonic velocity profile is set as an initial condition, along with magnetic fields. The flow sharpens into shocks and the simulation tracks the interaction between the magnetized shocks. Thin current sheets form and facilitate magnetic reconnection. The simulation is run to t=1.

Typically the problem is run with shock-capturing methods. Instead, here we regualrize the problem with viscosity, thermal diffusivity, and magnetic resistivity to regularize the problem. All diffusivities are equal (i.e., Pr=Pm=1). The shock thickness is related to the size of the diffusivities. For an accurate simulation, we require several grid points across the shock. Thus the minimum admissible resolution is related to the diffusivities. This simulation is run with Re=1e4 and 4096^2 Fourier modes.

### Resolution ###

```
nx = 4096
ny = 4096
```

This script was run for 13 hours on 1024 ivy-bridge cores on the Pleiades supercomputer.

## To Run ##

1. on a system of sufficent size, run 

run_problem.sh

2. after success, run 

run_analysis.sh

**This should produce figure <<X>> in Burns et al (2018).**

## Reference ##

Author, I. M. "Title of Paper", Journal of Great Results (2018) vol. 1, p.1-10


