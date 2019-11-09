#!/usr/bin/env bash

mpiexec python3 simulation.py
mpiexec python3 -m dedalus merge_procs snapshots --cleanup
mpiexec python3 -m dedalus merge_procs slices --cleanup
mpiexec python3 -m dedalus merge_procs traces --cleanup
