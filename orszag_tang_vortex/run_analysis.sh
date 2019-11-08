#!/usr/bin/env bash

mpiexec -n 1 python3 plot_density_profile.py
mpiexec -n 1 python3 plot_density_slice.py

