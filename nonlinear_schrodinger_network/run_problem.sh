#!/usr/bin/env bash

mpiexec -n 1 python3 simulation.py
mpiexec python3 -m dedalus merge_procs snap_sparse --cleanup
mpiexec python3 -m dedalus merge_procs snap_dense --cleanup

