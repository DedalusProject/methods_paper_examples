#!/usr/bin/env bash

python3 simulation.py
mpiexec python3 -m dedalus merge_procs snap_sparse --cleanup
mpiexec python3 -m dedalus merge_procs snap_dense --cleanup

