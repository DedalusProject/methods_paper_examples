#!/usr/bin/env bash
# takes approximately 13 hours on 1024 ivy-bridge cores on Pleiades.

mpirun -np 1024 python3 orszag_tang.py
python3 -m dedalus merge_procs snapshots_Re1e4_4096
