# Gravity Waves  #


## Problem Statement ##

Linear acoustic and gravity waves are a classic problem in fluid dynamics.  In constant coefficient systems (like isothermal atmospheres), exact analytic solutions exist.  In non constant coefficient systems, numerical techniques are a necessity.
This problem combines nonlinear boundary value problem (NLBVP) techniques with eigenvalue problem (EVP) techniques to solve for linearized acoustic and gravity waves in a non-trivial, astrophysically interesting atmosphere.  
Here we use radiative transfer to solve for an atmosphere in a NLBVP that spans regimes that are opaque to light and transparent to light, acting as a simple model of the transition between the interior of a star like the Sun and it's external atmosphere.  
Then we used Eigentools to solve pairs of dense eigenvalue problems to obtain all oscillating wave modes and to automatically reject numerically spurious solutions.
The resulting wave modes are classified based on asymptotic solutions.

### Resolution ###

```
kx = 20 logrithmically spaced modes
nz = 384
```

This test runs on 1 core and should run in approximately 6 hours on a Skylake machine.

## To Run ##

1. on a system of sufficent size, run

run_problem.sh

2. after success, run

run_analysis.sh

**This should produce figures 15, 16 and 17 in Burns et al (2019).**

## Reference ##

- Hindman, B. H. & Zweibel, E. G., "The effects of a hot outer atmosphere on acoustic-gravity waves", The Astrophysical Journal (1994) vol. 436, p.929-940
- Brown, B. P., Vasil, G. M, & Zweibel, E. G. "Energy conservation and gravity waves in sound-proof treatments of stellar interiors. Part I. Anelastic approximations", The Astrophysical Journal (2012) vol. 756, p.109:20pp
- Vasil, G. M, Lecoanet, D., Brown, B. P., Wood, T. S., & Zweibel, E. G. "Energy conservation and gravity waves in sound-proof treatments of stellar interiors. Part II. Lagrangian Constrained Analysis", The Astrophysical Journal (2013) vol. 773, p.169:23pp
