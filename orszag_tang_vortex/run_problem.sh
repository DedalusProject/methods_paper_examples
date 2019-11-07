#!/usr/bin/env bash
# takes approximately 13 hours on 1024 ivy-bridge cores on Pleiades.

mpiexec python3 simulation.py
mpiexec python3 -m dedalus merge_procs snapshots_Re1e4_4096 --cleanup
