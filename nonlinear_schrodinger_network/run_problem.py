#!/usr/bin/env bash

python3 nonlinear_schrodinger_network.py
mpiexec python3 -m dedalus merge_procs snapshots
mpiexec python3 -m dedalus merge_procs outputs
