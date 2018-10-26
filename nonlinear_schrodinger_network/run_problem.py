#!/usr/bin/env bash

mpirun -np X python3 nonlinear_schrodinger_network.py
mpirun -np X python3 -m dedalus merge_procs snapshots
mpirun -np X python3 -m dedalus merge_procs outputs
