#!/usr/bin/env bash

mpirun -np 1024 python3 orszag_tang.py
python3 -m dedalus merge_procs snapshots_Re1e4_4096
python3 -m dedalus merge_procs outputs_Re1e4_4096
