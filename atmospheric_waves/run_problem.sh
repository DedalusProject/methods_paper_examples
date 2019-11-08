#!/usr/bin/env bash

mpiexec -n 1 python3 atmosphere.py
mpiexec -n 1 python3 gravity_waves.py
