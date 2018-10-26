#!/usr/bin/env bash

mpirun -np X python3 shear_thinning.py
mpirun -np X python3 -m dedalus merge_procs snapshots
mpirun -np X python3 -m dedalus merge_procs outputs
