#!/usr/bin/env bash

mpiexec -n 1 python3 solve_atmosphere.py
mpiexec -n 1 python3 solve_waves.py
