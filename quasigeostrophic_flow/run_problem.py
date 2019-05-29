#!/usr/bin/env bash

mpirun -np 1024 python3 quasi_geostrophic.py
mpirun -np 100 python3 -m dedalus merge_procs snapshots
mpirun -np 20 python3 -m dedalus merge_procs traces
